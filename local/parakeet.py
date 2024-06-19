from pathlib import Path
import glob
import os

import lhotse
import torchaudio


def parse_parakeet(audio_dir, manifest_dir, manifest_name, text_file, txt_norm):
    sess_prefix = Path(audio_dir).stem
    audios = glob.glob(os.path.join(audio_dir, "*.ogg"))
    sess2audio = {
        str(sess_prefix) + "_" + str(Path(x).stem).split("_")[-1]: x for x in audios
    }

    with open(text_file, "r") as f:
        lines = f.readlines()

    lines = [l.strip("\n") for l in lines]

    recordings = []
    supervisions = []
    for indx, line in enumerate(lines):
        c_speakers = list(
            dict.fromkeys([x for x in line.split(" ") if x in ["[S1]", "[S2]"]])
        )
        speakers = {k: [] for k in c_speakers}
        c_spk = None

        for word in line.split(" "):
            if word in ["[S1]", "[S2]"]:
                c_spk = word
                continue
            else:
                try:
                    speakers[c_spk].append(word)
                except:
                    import pdb

                    pdb.set_trace

        c_sess_id = str(sess_prefix) + "_" + str(indx)
        info = torchaudio.info(sess2audio[c_sess_id])
        duration = info.num_frames / info.sample_rate
        c_rec = lhotse.Recording(
            id=c_sess_id,
            sources=[lhotse.AudioSource("file", [0], sess2audio[c_sess_id])],
            sampling_rate=info.sample_rate,
            num_samples=info.num_frames,
            duration=duration,
        )
        recordings.append(c_rec)

        for spk in speakers.keys():
            c_words = " ".join(speakers[spk])
            c_words = txt_norm(c_words)
            if len(c_words) > 0:
                sup_id = c_sess_id + f"-{spk}"
                c_sup = lhotse.SupervisionSegment(
                    sup_id, c_sess_id, 0.0, duration, channel=[0], text=c_words
                )
                supervisions.append(c_sup)

    recordings = lhotse.RecordingSet(recordings)
    supervisions = lhotse.SupervisionSet(supervisions)

    recordings, supervisions = lhotse.fix_manifests(recordings, supervisions)

    supervisions.to_file(
        os.path.join(manifest_dir, f"parakeet-supervisions_{manifest_name}.jsonl.gz")
    )
    recordings.to_file(
        os.path.join(manifest_dir, f"parakeet-recordings_{manifest_name}.jsonl.gz")
    )

    return supervisions, recordings


if __name__ == "__main__":
    pass
