from copy import deepcopy
from meeteval.wer import mimo_word_error_rate, orc_word_error_rate


def normSegLST(seglst, txt_norm):
    # apply text normalization to seglst and drop empty utts
    out = []
    for utt in seglst:
        new_utt = deepcopy(utt)
        new_utt["words"] = txt_norm(utt["words"])
        if len(new_utt["words"]):
            out.append(new_utt)
    out = sorted(out, key=lambda x: x["start_time"])
    return out


class WERErrorAcc:
    def __init__(self, wer_func=mimo_word_error_rate):
        self.wer_func = wer_func
        self.reset()

    def __call__(self, ref, hyp):
        if len(hyp) == 0:
            self.length += sum([len(x["words"].split(" ")) for x in ref])
            self.deletions += sum([len(x["words"].split(" ")) for x in ref])
            return 1.0
        else:
            c_wer_stats = self.wer_func(ref, hyp)
        self.length += c_wer_stats.length
        self.insertions += c_wer_stats.insertions
        self.deletions += c_wer_stats.deletions
        self.substitutions += c_wer_stats.substitutions

        return c_wer_stats.error_rate

    def reset(self):
        self.length = 0
        self.insertions = 0
        self.deletions = 0
        self.substitutions = 0

    def get_total(self):
        return (self.insertions + self.deletions + self.substitutions) / self.length
