import random
import torch
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self, x_input_ids, x_attention_masks, y, tokenizerid, tokenizer):
        super().__init__()
        assert len(x_input_ids) == len(x_attention_masks) == len(y)
        assert y[0].dtype == torch.float32, "y only for binary classification with BCE"
        assert len(y[0].shape) == 1, "y only for binary classification with BCE"

        self.tokenizer_id = f"{tokenizer.__class__.__name__}({tokenizerid})"

        # to store all the data
        self._x_input_ids = x_input_ids
        self._x_attention_masks = x_attention_masks
        self._y = y

        # to store working part of data (e.g. few shot)
        self.x_input_ids = self._x_input_ids
        self.x_attention_masks = self._x_attention_masks
        self.y = self._y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x_input_ids[idx]
        xm = self.x_attention_masks[idx]
        y = self.y[idx]
        return x, xm, y

    def get_input_len(self):
        return len(self.x_input_ids[0])

    def few_shot(self, shots_per_class=None, seed=7):
        """
        Sets the DS into few shot mode

        Parameters
        ----------
        shots_per_class: number of samples per class (if None resets the dataset and all samples are used)
        """
        if shots_per_class is None: # reset to full data
            self.x_input_ids = self._x_input_ids
            self.x_attention_masks = self._x_attention_masks
            self.y = self._y
            return self

        self.x_input_ids = []
        self.x_attention_masks = []
        self.y = []

        indices = list(range(len(self._y)))
        random.Random(seed).shuffle(indices)   # always shuffled the same
        pos_shots = 0
        neg_shots = 0
        for i in indices:
            if self._y[i] >= 0.5:   # add positive sample
                if pos_shots < shots_per_class:
                    self.x_input_ids.append(self._x_input_ids[i])
                    self.x_attention_masks.append(self._x_attention_masks[i])
                    self.y.append(self._y[i])
                    pos_shots += 1
            else:   # add negative
                if neg_shots < shots_per_class:
                    self.x_input_ids.append(self._x_input_ids[i])
                    self.x_attention_masks.append(self._x_attention_masks[i])
                    self.y.append(self._y[i])
                    neg_shots += 1

            if pos_shots == neg_shots == shots_per_class:
                break

        assert len(self) == 2 * shots_per_class, f"Not enough samples for {shots_per_class}: POS {pos_shots}, NEG {neg_shots}"
        assert sum(self.y) == shots_per_class, f"Not enough samples for {shots_per_class}: POS {pos_shots}, NEG {neg_shots}"
        return self

    def invert_y_labels(self):
        origdata = self._y[0]

        for i in range(len(self.y)):
            self.y[i] = 1 - self.y[i]

        if origdata == self._y[0]:  # if the lists are not the same and original were not changed
            for i in range(len(self._y)):
                self._y[i] = 1 - self._y[i]
                
        return self

if __name__ == '__main__':
    from dataset.mrda_dataset import MRDADataset
    from dataset.rquet_dataset import RquetDataset
    from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
    from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
    import torch
    import numpy as np

    ds = SwDA_RQDataset_Balanced("val", "gpt2", input_length=10)
    ds.few_shot(5)
    ds.invert_y_labels()


    tokenizerids = [
        "gpt2",
        "bert-base-cased"
    ]

    inputlen = 160

    output = []

    for tid in tokenizerids:
        lens = []
        for p in ["val", "train", "test"]:
            # ds = MRDADataset(p, tid, input_length=inputlen)
            # ds = SarcasmV2Dataset(p, tid, input_length=inputlen)
            # ds = RquetDataset(p, tid, input_length=inputlen, left_context=False, right_context=False)
            ds = SwDA_RQDataset_Balanced(p, tid, input_length=inputlen, left_context=False, right_context=False)

            for x, xm, y in ds:
                textlen = torch.sum(xm).detach().cpu().item()
                assert textlen < inputlen
                lens.append(textlen)

        mx = np.max(lens)
        mn = np.mean(lens)
        md = np.median(lens)
        perc = np.percentile(lens, 90)
        output.extend([mx, mn, md, perc])
        print(f"{mx} max len of {tid}")
        print(f"{mn} mean len of {tid}")
        print(f"{md} median len of {tid}")
        print(f"{perc} 90 percentile len of {tid}")

    text = "\t".join([f"{o}" for o in output])
    print(text)
