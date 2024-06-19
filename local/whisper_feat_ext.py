from . import whisper
import torch


def whisper_feat_extr(audio, is_training=False, channel_avg=True):
    if channel_avg:
        audio = audio.mean(0)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        # mel = mel.unsqueeze(0)
    else:
        raise NotImplementedError
        mel = []
        for ch in range(audio.shape[0]):
            c_audio = whisper.pad_or_trim(audio[ch].flatten())
            c_mel = whisper.log_mel_spectrogram(c_audio)
            mel.append(c_mel)
        mel = torch.stack(mel)

    # TODO put optionally here specaugment
    return mel
