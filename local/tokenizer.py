from collections import OrderedDict
import numpy as np


class SOTWrapperTokenizer:
    def __init__(self, orig_tok, sot_sym="|", sot_style="kanda"):
        self.tokenizer = orig_tok
        self.sot_sym = sot_sym
        self.sot_style = sot_style

    def encode_sot_kanda(self, uttg):
        uttg = sorted(uttg, key=lambda x: x["start_time"])
        spk_dict = OrderedDict.fromkeys([x["speaker"] for x in uttg])
        # cat utterances from each speaker
        for utt in uttg:
            c_spk = utt["speaker"]
            if spk_dict[c_spk] is None:
                spk_dict[c_spk] = []
            spk_dict[c_spk].append(utt)

        text = ""
        for indx, spk in enumerate(spk_dict.keys()):
            for utt in spk_dict[spk]:
                text += utt["words"] + " "
            text = text[:-1]
            if indx <= (len(spk_dict.keys()) - 2):
                text += " {} ".format(self.sot_sym)

                # text = text[:-1]
        return self.tokenizer.encode(text)

    def encode_sot_jordan(self, uttg, is_training=False):
        uttg = sorted(uttg, key=lambda x: x["start_time"])
        spk_dict = list(OrderedDict.fromkeys([x["speaker"] for x in uttg]))
        if is_training:
            np.random.shuffle(spk_dict)

        assert len(spk_dict) <= 2
        spk_dict = {x: f"[S{indx+1}]" for indx, x in enumerate(spk_dict)}
        # spk_dict = {spk_dict[0]: "[S1]", spk_dict[1]: "[S2]"}

        text = ""
        prev_spk = None
        for indx, _ in enumerate(uttg):
            c_utt = uttg[indx]
            if prev_spk is None:
                text += spk_dict[c_utt["speaker"]] + " "
            elif prev_spk != c_utt["speaker"]:
                text += spk_dict[c_utt["speaker"]] + " "
            else:
                pass
                # text += spk_dict[c_utt["speaker"]] + " "

            text += c_utt["words"] + " "
        text = text[:-1]
        return self.tokenizer.encode(text)

    def decode_sot_jordan(self, tokens):
        if isinstance(tokens, str):
            decoded = tokens
        else:
            decoded = self.tokenizer.decode(tokens)

        spk2utt = {}
        c_spk = None
        for word in decoded.split(" "):
            if word in ["[S1]", "[S2]"]:
                c_spk = word
                if c_spk not in spk2utt.keys():
                    spk2utt[c_spk] = []
            else:
                try:
                    spk2utt[c_spk].append(word)
                except:
                    print("failed decoding, no speaker at beginning ! falling back")
                    c_spk = "[S1]"
                    spk2utt[c_spk] = []
                    spk2utt[c_spk].append(word)

        output = []
        for spk in spk2utt.keys():
            output.append(" ".join(spk2utt[spk]).strip())
        return output

    def encode_sot_slidar(self, uttg):
        # more speaker changes, it is univoque in 2-spk setting
        uttg = sorted(uttg, key=lambda x: x["start_time"])
        # ordering is based on utterance start and stop
        c_spk = None
        text = ""
        for indx, _ in enumerate(uttg):
            c_utt = uttg[indx]
            if c_spk is None:
                c_spk = c_utt["speaker"]
            else:
                if c_spk != c_utt["speaker"]:
                    text += " {} ".format(self.sot_sym)
                    c_spk = c_utt["speaker"]

            text += c_utt["words"] + " "

        text = text[:-1]
        return self.tokenizer.encode(text)

    def encode_sot(self, uttg, is_training):
        if self.sot_style == "kanda":
            return self.encode_sot_kanda(uttg)
        elif self.sot_style == "slidar":
            return self.encode_sot_slidar(uttg)
        elif self.sot_style == "jordan":
            return self.encode_sot_jordan(uttg, is_training)
        else:
            raise NotImplementedError

    def decode_sot(self, uttg):
        if self.sot_style == "kanda":
            return self.decode_sot_kanda(uttg)
        elif self.sot_style == "slidar":
            return self.decode_sot_slidar(uttg)
        elif self.sot_style == "jordan":
            return self.decode_sot_jordan(uttg)
        else:
            raise NotImplementedError

    def decode_sot_kanda(self, tokens):
        decoded = self.tokenizer.decode(tokens)

        words = decoded.split(" ")
        if len(words) == 0:
            return words

        out = []
        stack = [words[0]]
        for word in words[1:]:
            if word == self.sot_sym:
                out.append([x for x in stack if x != ""])
                stack = []
            else:
                stack.append(word)

        if len(stack) > 0:
            out.append([x for x in stack if x != ""])
        return out

    def decode_sot_slidar(self, tokens):
        decoded = self.decode_sot_kanda(tokens)

        # even position == spk 0
        # odd == spk 1

        spk0 = []
        spk1 = []
        for indx in range(len(decoded)):
            if indx % 2 == 0:
                spk0.extend(decoded[indx])
            else:
                spk1.extend(decoded[indx])

        return [" ".join(spk0), " ".join(spk1)]

    def to_seglst(self, decoded, session_id, start_time=0.0, end_time=1.0):
        out = []
        for indx, spk_utt in enumerate(decoded):
            if self.sot_style == "jordan":
                out.append(
                    {
                        "speaker": str(indx),
                        "words": spk_utt,
                        "session_id": session_id,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )

            else:
                out.append(
                    {
                        "speaker": str(indx),
                        "words": " ".join(spk_utt),
                        "session_id": session_id,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
        return out
