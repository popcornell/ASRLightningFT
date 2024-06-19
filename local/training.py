import logging
import os
from functools import partial
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as torch_dist
from speechbrain.nnet.losses import kldiv_loss
from torchmetrics.classification.accuracy import MulticlassAccuracy
from .batching import PaddedBatch
from .padding import batch_pad_right
from transformers import AdamW, get_linear_schedule_with_warmup
from .scoring import WERErrorAcc, normSegLST
from meeteval.wer import mimo_word_error_rate, orc_word_error_rate, cp_word_error_rate
from chime_utils.text_norm.whisper_like.english import EnglishTextNormalizer
import json

log = logging.getLogger(__name__)


TXT_NORM = EnglishTextNormalizer(
    standardize_numbers=True, standardize_numbers_rev=False
)


def get_worker_rand_seed(worker_num):
    return np.random.seed(int.from_bytes(os.urandom(4), "little") + worker_num)


MultiDimPaddedBatch = partial(PaddedBatch, padding_func=batch_pad_right)


class ASRModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        tr_ds,
        cv_ds,
        tt_ds,
        asr_model,
        tokenizer,
        asr_options,
    ):
        super(ASRModule, self).__init__()
        self.hparams.update(cfg)
        self.tr_ds = tr_ds
        self.cv_ds = cv_ds
        self.tt_ds = tt_ds
        self.asr_model = asr_model
        self.tokenizer = tokenizer
        self.asr_options = asr_options

        if self.hparams.debug:
            self.num_workers = 1
        else:
            self.num_workers = self.hparams.training.num_workers

        for split in ["val", "train"]:
            for acc_name in ["overall"]:
                setattr(
                    self,
                    f"acc_{split}_{acc_name}",
                    MulticlassAccuracy(
                        self.asr_model.dims.n_vocab, average="micro", ignore_index=0
                    ),
                )

        self.nan_countdown = 1
        self.tr_its_epoch = len(tr_ds) // self.hparams.training.batch_size

        self.orcwer = WERErrorAcc(orc_word_error_rate)
        self.mimower = WERErrorAcc(mimo_word_error_rate)
        self.cpwer = WERErrorAcc(cp_word_error_rate)

        self.to_csv = []

    def _sync2skip(self, flag_skip):
        world_size = torch_dist.get_world_size()
        torch_dist.barrier()
        # now gather
        result = [torch.zeros_like(flag_skip) for _ in range(world_size)]
        torch_dist.all_gather(result, flag_skip)
        any_invalid = torch.sum(torch.stack(result)).bool().item()
        return any_invalid

    def _reduce_loss(self, utt_id, batch_id, loss, reduction="mean"):
        assert loss.shape[0] == len(utt_id), "loss must be reduced to batch dimension !"

        mask_nan_inf = torch.logical_or(torch.isnan(loss), ~torch.isfinite(loss))
        if torch.any(mask_nan_inf):
            where_invalid = torch.where(mask_nan_inf)[0]
            for indx in range(where_invalid.shape[0]):
                inv_indx = where_invalid[indx].item()
                log.info(
                    f"NaN loss in batch {batch_id} of epoch {self.current_epoch}, "
                    f"for utt_id {utt_id[inv_indx]}"
                )
            # if any is invalid then we must flag this to all processes
            flag_skip = torch.ones((), device=loss.device, dtype=torch.bool)
        else:
            flag_skip = torch.zeros((), device=loss.device, dtype=torch.bool)

        # sub-optimal but will do,
        # till they fix it in https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1552650013
        any_invalid = self._sync2skip(flag_skip)

        if any_invalid:
            if self.nan_countdown >= 100:
                raise RuntimeError(
                    "Too many NaNs loss iterations encountered, stopping !"
                )
            self.nan_countdown += 1
        else:
            self.nan_countdown = 1

        return (
            loss.mean() if reduction == "mean" else loss.sum(),
            ~mask_nan_inf,
            any_invalid,
        )

    def _log_accuracy(self, decoder_out, tokens_eos, split="train"):
        for acc_name in ["overall"]:
            c_target = tokens_eos

            acc_obj = getattr(self, f"acc_{split}_{acc_name}")

            acc_obj(
                decoder_out.view(-1, self.asr_model.dims.n_vocab), c_target.reshape(-1)
            )

            self.log(
                f"{split}/accuracy_{acc_name}",
                acc_obj,
                on_step=True if split == "train" else False,
                on_epoch=True,
                prog_bar=True if acc_name == "overall" else False,
                batch_size=c_target.shape[0],
            )

    def training_step(self, batch, batch_id):
        audio, wav_lens = batch.waveform
        batch_size = audio.shape[0]
        utt_id = batch.utt_id

        # print(audio.shape[-1])

        wav_lens = wav_lens[:, -1]
        tokens_eos, teos_lens = batch.tgt_eos
        tokens_bos, tbos_lens = batch.tgt_bos

        audio_features = self.asr_model.encoder(audio)
        decoder_out = self.asr_model.decoder(
            tokens_bos, audio_features, tbos_lens[:, 0]
        )

        with torch.cuda.amp.autocast(enabled=False):
            loss_kdiv = kldiv_loss(
                torch.log_softmax(decoder_out.float(), -1),
                tokens_eos,
                teos_lens[:, 0],
                pad_idx=0,
                allowed_len_diff=0,
                # label_smoothing=self.hparams.training.label_smoothing, #TODO check
                reduction="batch",
            )

        loss = loss_kdiv
        loss, valid_mask, has_nans = self._reduce_loss(utt_id, batch_id, loss, "mean")

        # in early iterations, we avoid
        if has_nans:
            return None

        self._log_accuracy(decoder_out, tokens_eos, split="train")

        self.log(
            "train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        self.log(
            "train/lr",
            self.optimizer.param_groups[-1]["lr"],
            prog_bar=True,
            on_step=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_id):
        audio, wav_lens = batch.waveform
        utt_id = batch.utt_id

        wav_lens = wav_lens[:, -1]
        tokens_eos, teos_lens = batch.tgt_eos
        tokens_bos, tbos_lens = batch.tgt_bos

        audio_features = self.asr_model.encoder(audio)
        decoder_out = self.asr_model.decoder(
            tokens_bos, audio_features, tbos_lens[:, 0]
        )

        loss_kdiv = kldiv_loss(
            torch.log_softmax(decoder_out, -1),
            tokens_eos,
            teos_lens[:, 0],
            pad_idx=0,
            allowed_len_diff=0,
            # label_smoothing=self.hparams.training.label_smoothing, #TODO check
            reduction="batch",
        )

        loss = loss_kdiv
        loss, valid_mask, has_nans = self._reduce_loss(utt_id, batch_id, loss, "mean")

        # in early iterations, we avoid
        if has_nans:
            return None

        self._log_accuracy(decoder_out, tokens_eos, split="val")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=audio.shape[0],
        )

        return

    def test_step(self, batch, batch_id):
        audio, wav_lens = batch.waveform
        utt_id = batch.utt_id
        batch_size = audio.shape[0]

        text = batch.text

        with torch.cuda.amp.autocast(enabled=False):
            result = self.asr_model.decode(audio.float(), options=self.asr_options)

        for b_indx in range(batch_size):
            hyp = self.tokenizer.decode_sot(result[b_indx].tokens)
            hyp = self.tokenizer.to_seglst(hyp, text[b_indx][0]["session_id"])
            ref = text[b_indx]
            hyp = normSegLST(hyp, TXT_NORM)
            ref = normSegLST(ref, TXT_NORM)

            self.to_csv.append(
                {
                    "ref": ref,
                    "hyp": hyp,
                    "dec_text": result[b_indx].text,
                    "dec_tokens": result[b_indx].tokens,
                    "target": self.tokenizer.tokenizer.decode(batch.tgt_bos[b_indx][0]),
                }
            )

            try:
                # self.orcwer(ref, hyp)
                # self.mimower(ref, hyp)
                self.cpwer(ref, hyp)
            except:
                import pdb

                pdb.set_trace()

        print(self.cpwer.get_total())
        # print( self.orcwer.get_total())
        print("****************")

        self.log(
            "test/cp_wer",
            self.cpwer.get_total(),
            on_step=False,
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=audio.shape[0],
        )

        return

    def on_test_epoch_end(self):
        with open("./preds.json", "w") as f:
            json.dump(self.to_csv, f, indent=4)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.t_total = self.tr_its_epoch * float(self.hparams.training.max_epochs)
            self.t_total = self.t_total // (
                self.hparams.training.gradient_accumulation_steps * self.hparams.gpus
            )

    def configure_optimizers(self):
        if self.hparams.opt.type == "adam":
            raise NotImplementedError
            """
            trainable_parameters = [
                (n, p) for n, p in self.asr_model.named_parameters()
            ] + [
            #        (n, p)
            #        for n, p in self.teacher.named_parameters()
            #        if not n.startswith("model.frontend.hf_frontend")
                ] + [(n, p)
                    for n, p in self.pre_encoder.named_parameters()]

            # [
            # (n, p) for n, p in self.student.named_parameters()
            # ]
            """
            trainable_parameters = (
                [(n, p) for n, p in self.asr_model.named_parameters()]
                + [(n, p) for n, p in self.pre_encoder.named_parameters()]
                + [(n, p) for n, p in self.spk_vector_loss.named_parameters()]
            )
            trainable_parameters += [
                (n, p)
                for n, p in self.teacher.named_parameters()
                if not n.startswith("model.frontend.hf_frontend")
            ]

            if self.hparams.wavlm.freeze:
                opt_grouped_params = [
                    {
                        "params": [x[-1] for x in trainable_parameters],
                        "lr": self.hparams.opt.learning_rate,
                    }
                ]
            else:
                trainable_parameters_wavlm = [
                    (n, p)
                    for n, p in self.teacher.named_parameters()
                    if n.startswith("model.frontend.hf_frontend")
                ]

                # import pdb
                # pdb.set_trace()

                opt_grouped_params = [
                    {
                        "params": [x[-1] for x in trainable_parameters],
                        "lr": self.hparams.opt.learning_rate,
                    },
                    {
                        "params": [x[-1] for x in trainable_parameters_wavlm],
                        "lr": self.hparams.opt.learning_rate,
                    },
                ]

            optimizer = torch.optim.Adam(
                opt_grouped_params,
                lr=self.hparams.opt.learning_rate,
                eps=self.hparams.opt.adam_epsilon,
            )

        elif self.hparams.opt.type == "adamw":
            # lora = [
            #    "lora_B",
            #    "lora_A"
            #    "positional_embedding_extended",
            # ]
            trainable_parameters = []
            for n, p in self.asr_model.named_parameters():
                if p.requires_grad:
                    # if any([nd in n.split(".") for nd in lora]):
                    trainable_parameters.append(p)

            trainable_parameters = [
                {
                    "params": trainable_parameters,
                    "weight_decay": self.hparams.opt.weight_decay,
                },
            ]

            print(
                "Optimizer Parameters: {} M".format(
                    sum([p.numel() for p in trainable_parameters[0]["params"]]) / 1e6
                )
            )
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.hparams.opt.learning_rate,
                eps=self.hparams.opt.adam_epsilon,
            )

        else:
            # adamw maybe ?
            raise NotImplementedError
        self.optimizer = optimizer
        if self.hparams.opt.scheduler == "warmup":
            warmup_steps = self.hparams.warmup.warmup_steps * self.tr_its_epoch
            warmup_steps = warmup_steps // (
                self.hparams.training.gradient_accumulation_steps * self.hparams.gpus
            )
            # ( self.hparams.training.batch_size
            # // self.hparams.training.gradient_accumulation_steps)

            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.t_total,
            )
            # these are handled by lightning
            log.info(f"Warmup Steps: {warmup_steps}")
            self.scheduler = scheduler

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif self.hparams.opt.scheduler == "reduce":
            scheduler = [
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=self.hparams.reduce.reduce_f,
                        patience=self.hparams.reduce.patience,
                        verbose=True,
                    ),
                    "monitor": "val/loss",
                }
            ]
            self.scheduler = scheduler
            return [optimizer], scheduler
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.tr_ds,
            batch_size=self.hparams.training.batch_size,
            num_workers=self.hparams.training.num_workers,
            shuffle=True,
            worker_init_fn=partial(get_worker_rand_seed),
            collate_fn=MultiDimPaddedBatch,
            drop_last=True,  # might be needed for DDP
        )

    def val_dataloader(self):
        batch_size = self.hparams.training.batch_size
        return torch.utils.data.DataLoader(
            self.cv_ds,
            batch_size=batch_size,
            num_workers=self.hparams.training.num_workers,
            shuffle=False,
            collate_fn=MultiDimPaddedBatch,
            drop_last=True,  # might be needed for DDP
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.tt_ds,
            batch_size=1,
            num_workers=self.hparams.training.num_workers,
            shuffle=False,
            collate_fn=MultiDimPaddedBatch,
            drop_last=True,  # might be needed for DDP
        )
