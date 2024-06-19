import os.path

from local import whisper
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from local.tokenizer import SOTWrapperTokenizer

from local.training import ASRModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import torchaudio
from data_init import init_dataio

# /raid/users/popcornell/SynthSpeech/egs/fisher/whisper_ft_lightning/outputs/2024-06-16/10-28-38/lightning_logs


@hydra.main(config_path="conf", config_name="whisper_ft")
def single_run(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.random.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torchaudio.set_audio_backend("soundfile")

    woptions = whisper.DecodingOptions(
        language="en", without_timestamps=True, beam_size=cfg.whisper.beam_size
    )
    asr_model = whisper.load_model(
        cfg.whisper.model,
        device="cpu",
        load_strict=False,
    )

    if cfg.whisper.lora:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

        rank = 64
        config = LoraConfig(
            r=rank,
            lora_alpha=128,  # (rank)**0.5,
            target_modules=["query", "value", "key", "mlp.0", "mlp.2"],
            lora_dropout=0.1,
            bias="none",
        )

        asr_model = get_peft_model(asr_model, config)
        asr_model.print_trainable_parameters()
        # asr_model.decoder.positional_embedding_extended.requires_grad = True
        asr_model.print_trainable_parameters()
        asr_model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

    tokenizer = SOTWrapperTokenizer(
        whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task),
        cfg.tokenizer.sot_sym,
        cfg.tokenizer.sot_style,
    )

    tr_ds, cv_ds, tt_ds = init_dataio(cfg, tokenizer)
    asr_module = ASRModule(cfg, tr_ds, cv_ds, tt_ds, asr_model, tokenizer, woptions)
    exp_logger = TensorBoardLogger(
        os.getcwd(),
    )

    callbacks = [
        EarlyStopping(
            monitor="val/accuracy_overall",
            patience=cfg.training.early_stop_patience,
            verbose=True,
            mode="max",
        ),
        ModelCheckpoint(
            exp_logger.log_dir,
            monitor="val/accuracy_overall",
            save_top_k=1,
            mode="max",
            save_last=True,
        ),
    ]

    trainer = pl.Trainer(
        precision=cfg.training.precision,
        max_epochs=cfg.training.max_epochs,
        callbacks=callbacks,
        devices=cfg.gpus,
        accelerator="gpu",
        use_distributed_sampler=True,
        strategy=cfg.training.strategy,
        accumulate_grad_batches=cfg.training.gradient_accumulation_steps,
        logger=exp_logger,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        gradient_clip_val=cfg.training.gradient_clip,
        check_val_every_n_epoch=cfg.training.validation_interval,
        num_sanity_val_steps=0,
        fast_dev_run=True if cfg.debug else False,
    )

    if cfg.test_only:
        # only run test with unadapted model.
        best_metric = trainer.test(asr_module, ckpt_path=None)
        print("RESULTS IN {}".format(os.getcwd()))
        return best_metric

    if cfg.test_from_checkpoint is None:
        trainer.fit(asr_module, ckpt_path=cfg.resume_from_checkpoint)
        checkpoint_path = "best"
    else:
        checkpoint_path = cfg.test_from_checkpoint
    best_metric = trainer.test(asr_module, ckpt_path=checkpoint_path)
    print("RESULTS IN {}".format(os.getcwd()))
    return best_metric


if __name__ == "__main__":
    single_run()
