from chime_utils.text_norm.whisper_like.english import EnglishTextNormalizer
from functools import partial
from local.utils import run_stage
from local.whisper_feat_ext import whisper_feat_extr
from local.fisher import fisher2lhotse
from local.dataio import ASRUttGDataset
import torch
import os


def init_dataio(cfg, tokenizer):
    run_stage_flag = partial(
        run_stage,
        start_stage=cfg.stage,
        stop_stage=cfg.stop_stage,
        skip_stages=cfg.skip_stages,
    )

    if cfg.data.txt_norm == "whisper":
        txt_norm = EnglishTextNormalizer(
            standardize_numbers=True, standardize_numbers_rev=False
        )
    elif cfg.data.txt_norm in [None, False]:
        txt_norm = lambda x: x
    else:
        raise NotImplementedError

    if run_stage_flag(0):
        if cfg.data.train_set == "fisher" or cfg.data.test_set == "fisher":
            # parse fisher
            for split in [
                "training_set_p1",
                "training_set_p2",
                "validation_set",
                "test_set",
            ]:
                if split == "training_set_p1":
                    audio_path = os.path.join(
                        cfg.data.fisher.audio_dir, "training_set", "mix"
                    )
                else:
                    audio_path = os.path.join(cfg.data.fisher.audio_dir, split, "mix")
                fisher2lhotse(
                    audio_path,
                    os.path.join(cfg.data.fisher.csv_dir, split + ".csv"),
                    split,
                    cfg.data.fisher.manifest_dir,
                    txt_norm,
                )

        else:
            pass
            # raise NotImplementedError

    if cfg.data.train_set == "fisher":
        tr_ds = []
        for dset in cfg.data.fisher.tr_dsets:
            c_sup = os.path.join(
                cfg.data.fisher.manifest_dir, f"fisher-supervisions_{dset}.jsonl.gz"
            )
            c_rec = os.path.join(
                cfg.data.fisher.manifest_dir, f"fisher-recordings_{dset}.jsonl.gz"
            )

            tmp = ASRUttGDataset(
                c_sup,
                c_rec,
                tokenizer,
                orig_samplerate=8000,
                tgt_samplerate=16000,
                sim_telephone_speech=False,
                is_training=True,
                discard_longer=cfg.data.discard_longer,
                discard_shorter=cfg.data.discard_shorter,
                feat_ext=whisper_feat_extr,
            )
            tmp[0]
            tr_ds.append(tmp)

        tr_ds = tr_ds[0]  # torch.utils.data.ConcatDataset(tr_ds)

    elif cfg.data.train_set == "nemo":
        nemo_subset = "train_clean"
        c_sup = os.path.join(
            cfg.data.nemo.manifest_dir,
            f"{nemo_subset}",
            f"{nemo_subset}_supervisions.jsonl.gz",
        )
        c_rec = os.path.join(
            cfg.data.nemo.manifest_dir,
            f"{nemo_subset}",
            f"{nemo_subset}_recordings.jsonl.gz",
        )

        tr_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            limit_ds_ex=cfg.data.nemo.limit_ex,
            orig_samplerate=16000,
            tgt_samplerate=16000,
            sim_telephone_speech=True if cfg.data.test_set == "fisher" else False,
            is_training=True,
            discard_longer=cfg.data.discard_longer,
            discard_shorter=cfg.data.discard_shorter,
            feat_ext=whisper_feat_extr,
            max_pause_uttg=1.0,
        )
        tr_ds[0]

    elif cfg.data.train_set == "mixer6":
        c_sup = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_supervisions_train.jsonl.gz"
        )
        c_rec = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_recordings_train.jsonl.gz"
        )
        tr_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            orig_samplerate=16000,
            tgt_samplerate=16000,
            sim_telephone_speech=False,  # False if cfg.data.train_set != "fisher" else True,
            # test fisher used as training set for mixer6
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )
        tr_ds[0]

    elif cfg.data.train_set == "xtts":
        from local.coqui import CoquiDataset

        c_sup = os.path.join(
            "/raid/users/popcornell/SynthSpeech/egs/fisher/whisper_ft_lightning/xTTS_large_extra/xtts_cutset_combined.jsonl.gz"
        )
        tr_ds = CoquiDataset(
            c_sup,
            tokenizer,
            orig_samplerate=24000,
            tgt_samplerate=16000,
            #limit_ds_ex=3000,
            sim_telephone_speech=True,  # False if cfg.data.train_set != "fisher" else True,
            # test fisher used as training set for mixer6
            #rir_augmentation="/raid/users/popcornell/SynthSpeech/egs/fisher/whisper_ft_lightning/rir_noise/RIRS_NOISES/simulated_rirs",
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )
        tr_ds[0]

    else:
        raise NotImplementedError

    if cfg.data.test_set == "fisher":
        c_sup = os.path.join(
            cfg.data.fisher.manifest_dir, f"fisher-supervisions_validation_set.jsonl.gz"
        )
        c_rec = os.path.join(
            cfg.data.fisher.manifest_dir, f"fisher-recordings_validation_set.jsonl.gz"
        )
        cv_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            orig_samplerate=8000,
            tgt_samplerate=16000,
            sim_telephone_speech=False,
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )

        c_sup = os.path.join(
            cfg.data.fisher.manifest_dir,
            f"fisher-supervisions_test_set.jsonl.gz",
        )
        c_rec = os.path.join(
            cfg.data.fisher.manifest_dir, f"fisher-recordings_test_set.jsonl.gz"
        )
        tt_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            orig_samplerate=8000,
            tgt_samplerate=16000,
            sim_telephone_speech=False,
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )
    elif cfg.data.test_set == "mixer6":
        c_sup = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_supervisions_dev.jsonl.gz"
        )
        c_rec = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_recordings_dev.jsonl.gz"
        )
        cv_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            orig_samplerate=16000,
            tgt_samplerate=16000,
            sim_telephone_speech=False,  # False if cfg.data.train_set != "fisher" else True,
            # test fisher used as training set for mixer6
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )
        cv_ds[0]

        c_sup = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_supervisions_dev.jsonl.gz"
        )
        c_rec = os.path.join(
            cfg.data.mixer6.manifest_dir, "mixer6-mdm_recordings_dev.jsonl.gz"
        )
        tt_ds = ASRUttGDataset(
            c_sup,
            c_rec,
            tokenizer,
            orig_samplerate=16000,
            tgt_samplerate=16000,
            sim_telephone_speech=False,  # False if cfg.data.train_set != "fisher" else True,
            is_training=False,
            discard_longer=30,
            discard_shorter=-1,
            feat_ext=whisper_feat_extr,
        )
        tt_ds[0]

    else:
        raise NotImplementedError

    import numpy as np

    dur_tr = [x.duration for x in tr_ds.cuts]
    dur_tt = [x.duration for x in tt_ds.cuts]
    print("Mean duration {} train {} test".format(np.mean(dur_tr), np.mean(dur_tt)))
    print("Tot duration train set {} h".format(np.sum(dur_tr) / 3600))

    import pdb

    pdb.set_trace()

    if cfg.data.train_set == "nemo" and cfg.data.test_set == "mixer6":
        raise NotImplementedError

    return tr_ds, cv_ds, tt_ds
