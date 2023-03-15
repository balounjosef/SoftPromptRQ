import torch
from dataset.abstract_ds import AbstractDataset
from models.utils.model_utils import decide_tokenizer
from rquet_experiments import rquet_data_loader


# https://aclanthology.org/2021.iwcs-1.13.pdf
class RquetDataset(AbstractDataset):
    def __init__(self, part, tokenizerid, input_length=128-10, left_context=False, right_context=False):
        """
        Parameters
        ----------
        part: part of the dataset "test" or "train"
        tokenizerid: e.g. "gpt2" or "microsoft/DialoGPT-small"
        input_length: length of text input sequence (test max 164 tokens, train max 211)
        """
        assert part in ["test", "val", "train"]
        tokenizer, sep_text, last_token = decide_tokenizer(tokenizerid)

        data = rquet_data_loader.load_rquet_data(part)
        x_input_ids = []
        x_attention_masks = []
        y = []
        for d in data:
            input_txt = d.question
            if left_context:
                input_txt = f"{d.ctx_before2} {d.ctx_before1}{sep_text}" + input_txt
            if right_context:
                input_txt = input_txt + f"{sep_text}{d.ctx_after1} {d.ctx_after2}"
            input_txt = input_txt + last_token

            x = tokenizer(input_txt, padding="max_length", truncation=True, max_length=input_length, return_tensors="pt")
            x_input_ids.append(x.input_ids[0])
            x_attention_masks.append(x.attention_mask[0])
            if d.gold_label == "ISQ":
                y.append(torch.tensor([0], dtype=torch.float32))
            else:
                y.append(torch.tensor([1], dtype=torch.float32))

        super().__init__(x_input_ids=x_input_ids, x_attention_masks=x_attention_masks, y=y, tokenizerid=tokenizerid, tokenizer=tokenizer)

