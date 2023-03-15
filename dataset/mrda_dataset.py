import torch
from torch.utils.data import Dataset

from dataset.abstract_ds import AbstractDataset
from models.utils.model_utils import decide_tokenizer
from mrda_experiments import mrda_data_loader


class MRDADataset(AbstractDataset):
    def __init__(self, part, tokenizerid, input_length=128-10):
        assert part in ["test", "val", "train"]
        tokenizer, sep_text, last_token = decide_tokenizer(tokenizerid)

        data = mrda_data_loader.load_mrda_data(part)
        x_input_ids = []
        x_attention_masks = []
        y = []
        for d in data:
            if d.basic_label == "Q":    # only questions filtered
                input_txt = f"{d.utterance}{last_token}"
                x = tokenizer(input_txt, padding="max_length", truncation=True, max_length=input_length, return_tensors="pt")
                x_input_ids.append(x.input_ids[0])
                x_attention_masks.append(x.attention_mask[0])
                if d.general_label == "qh": # rhetorical as 1, otherwise 0
                    y.append(torch.tensor([1], dtype=torch.float32))
                else:
                    y.append(torch.tensor([0], dtype=torch.float32))

        super().__init__(x_input_ids=x_input_ids, x_attention_masks=x_attention_masks, y=y, tokenizerid=tokenizerid, tokenizer=tokenizer)


class MRDADataset_Balanced(MRDADataset):
    def __init__(self, part, tokenizerid, input_length=128-10):
        assert part in ["test", "val", "train"]
        super().__init__(part=part, tokenizerid=tokenizerid, input_length=input_length)
        # test  POS 51, NEG 1180
        # val   POS 33, NEG 1079
        # train POS 238, NEG 4402

        if part == "test":
            self.few_shot(shots_per_class=51)
        elif part == "val":
            self.few_shot(shots_per_class=33)
        elif part == "train":
            self.few_shot(shots_per_class=238)

        # fix default data to balanced
        self._x_input_ids = self.x_input_ids
        self._x_attention_masks = self.x_attention_masks
        self._y = self.y


if __name__ == '__main__':
    ds = MRDADataset_Balanced("val", "gpt2")
    ds = MRDADataset("val", "gpt2")

