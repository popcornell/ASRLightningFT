import glob
import pandas as pd
import os
import torchaudio
from pathlib import Path
import lhotse


def fisher2lhotse(audio_dir, csv_file, manifest_name, manifest_dir, txt_norm):
    # parsing fisher to lhotse manifests
    os.makedirs(manifest_dir, exist_ok=True)
    # parse audio into recordings.
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    recordings = []
    id2rec = {}
    for audio_file in audio_files:
        info = torchaudio.info(audio_file)
        source = lhotse.AudioSource(type="file", channels=[0], source=audio_file)
        c_rec = lhotse.Recording(
            id=Path(audio_file).stem,
            sources=[source],
            sampling_rate=info.sample_rate,
            num_samples=info.num_frames,
            duration=info.num_frames / info.sample_rate,
        )
        id2rec[Path(audio_file).stem] = c_rec
        recordings.append(c_rec)

    recordings = lhotse.RecordingSet(recordings)

    # now parse supervisions
    df = pd.read_csv(csv_file)
    sessions = df["filename"].unique().tolist()
    supervisions = []
    for c_sess in sessions:
        c_rec = id2rec[c_sess]
        # recording
        # segments in current session
        c_sess_utts = df[df["filename"].isin([c_sess])]

        for indx in range(len(c_sess_utts)):
            utt = c_sess_utts.iloc[indx]
            utt_id = "{}-{}-{}-{}".format(
                c_sess, str(utt["speaker"]), utt["start"], utt["end"]
            )
            # apply text normalization
            words = txt_norm(str(utt["transcription"]))

            if len(words) > 0 and (utt["end"] < (c_rec.duration)):
                # discard supervisions after end of recording
                c_sup = lhotse.SupervisionSegment(
                    utt_id,
                    recording_id=c_rec.id,
                    start=utt["start"],
                    duration=(utt["end"] - utt["start"]),
                    speaker=c_sess + "_" + str(utt["speaker"]),
                    text=words,
                )
                supervisions.append(c_sup)

    supervisions = lhotse.SupervisionSet(supervisions)

    recordings, supervisions = lhotse.fix_manifests(recordings, supervisions)

    supervisions.to_file(
        os.path.join(manifest_dir, f"fisher-supervisions_{manifest_name}.jsonl.gz")
    )
    recordings.to_file(
        os.path.join(manifest_dir, f"fisher-recordings_{manifest_name}.jsonl.gz")
    )
