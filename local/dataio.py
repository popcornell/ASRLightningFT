import torch
import torchaudio
import numpy as np
import lhotse


class ASRUttGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        supervisions,
        recordings,
        tokenizer,
        orig_samplerate,
        tgt_samplerate,
        sim_telephone_speech=False,
        is_training=False,
        discard_longer=200,
        discard_shorter=-1,
        limit_ds_ex=None,
        feat_ext=None,
        max_pause_uttg=0.0,
    ):
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.discard_longer = discard_longer
        self.discard_shorter = discard_shorter
        self.feat_ext = feat_ext
        self.orig_samplerate = orig_samplerate
        self.tgt_samplerate = tgt_samplerate
        self.sim_telephone_speech = sim_telephone_speech

        if not isinstance(supervisions, lhotse.SupervisionSet):
            supervisions = lhotse.load_manifest(supervisions)

        assert isinstance(supervisions, lhotse.SupervisionSet)
        if not isinstance(recordings, lhotse.RecordingSet):
            recordings = lhotse.load_manifest(recordings)

        assert isinstance(recordings, lhotse.RecordingSet)
        # here use cutset

        cutset = lhotse.CutSet.from_manifests(recordings, supervisions)

        cutset = cutset.trim_to_supervision_groups(
            max_pause=max_pause_uttg, num_jobs=12
        )
        prev_len = len(cutset)

        cutset = cutset.filter(lambda x: x.duration <= discard_longer).to_eager()
        print(
            f"Discarded {prev_len - len(cutset)} supervisions as they are longer than {discard_longer}. Before {prev_len}. Now {len(cutset)}"
        )

        if discard_shorter > 0:
            cutset = cutset.filter(lambda x: x.duration >= discard_shorter).to_eager()
            print(
                f"Discarded {prev_len - len(cutset)} supervisions as they are shorter than {discard_shorter}. Before {prev_len}. Now {len(cutset)}"
            )

        if limit_ds_ex is not None:
            # assert self.is_training
            cutset = cutset[:limit_ds_ex]
        self.cuts = cutset
        self.length = len(self.cuts)

        if self.sim_telephone_speech:
            if not self.is_training:
                nyquist_tel = 8000
            else:
                # random lowpass filter
                nyquist_tel = np.random.randint(6000, 8000)

            self.resample = torchaudio.transforms.Resample(
                self.orig_samplerate, nyquist_tel
            )
            self.resample_again = torchaudio.transforms.Resample(
                nyquist_tel, self.tgt_samplerate
            )
        else:
            if self.orig_samplerate != self.tgt_samplerate:
                self.resample = torchaudio.transforms.Resample(
                    self.orig_samplerate, self.tgt_samplerate
                )

    def __len__(self):
        return self.length

    def read_audio(self, c_cut):
        audio = c_cut.load_audio()
        audio = torch.Tensor(audio)

        assert c_cut.recording.sampling_rate == self.orig_samplerate
        if audio.shape[0] > 1:
            audio = audio[0].unsqueeze(0)
        # assert audio.shape[0] == 1
        # only mono for now

        if self.sim_telephone_speech:
            # 16000 -> 8000 -> 16000
            audio = self.resample(audio)
            audio = self.resample_again(audio)
            return audio

        if self.orig_samplerate != self.tgt_samplerate:
            audio = self.resample(audio)

        return audio

    def cut2segslt(self, c_cut):
        out = []
        for cut in c_cut.supervisions:
            out.append(
                {
                    "speaker": cut.speaker,
                    "start_time": cut.start,
                    "end_time": cut.duration + cut.start,
                    "words": cut.text,
                    "session_id": c_cut.recording.id,
                }
            )

        return sorted(out, key=lambda x: x["start_time"])

    def __getitem__(self, index):
        c_cut = self.cuts[index]
        # get start and end time

        waveform = self.read_audio(c_cut)
        if self.feat_ext is not None:
            # e.g. apply whisper feature extraction
            waveform = self.feat_ext(waveform, self.is_training)

        segslt = self.cut2segslt(c_cut)
        tgt = self.tokenizer.encode_sot(segslt, self.is_training)
        bos_seq = [*self.tokenizer.tokenizer.sot_sequence_including_notimestamps]

        return {
            "utt_id": c_cut.id,
            "waveform": waveform,
            "text": segslt,
            "tgt_eos": torch.tensor(
                [0] * (len(bos_seq) - 1) + tgt + [self.tokenizer.tokenizer.eot]
            ).long(),
            "tgt_bos": torch.tensor(bos_seq + tgt).long(),
            "cut": c_cut,
        }
