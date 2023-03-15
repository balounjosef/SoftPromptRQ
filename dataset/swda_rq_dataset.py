import torch
from dataset.abstract_ds import AbstractDataset
from models.utils.model_utils import decide_tokenizer
from swda_rq_experiments import swda_rq_data_loader


class SwDA_RQDataset_Balanced(AbstractDataset):
    def __init__(self, part, tokenizerid, input_length=128-10, left_context=False, right_context=False):
        if part == "test":
            part = "test_balanced"
        elif part == "val":
            part = "validate1_balanced"
        elif part == "train":
            part = "train1_balanced"

        assert part in ["test_balanced", "validate1_balanced", "train1_balanced", "all"]
        tokenizer, sep_text, last_token = decide_tokenizer(tokenizerid)

        if part == "all":
            data = swda_rq_data_loader.load_preprocessed_swda_data("test_balanced") + swda_rq_data_loader.load_preprocessed_swda_data("validate1_balanced") + swda_rq_data_loader.load_preprocessed_swda_data("train1_balanced")
        else:
            data = swda_rq_data_loader.load_preprocessed_swda_data(part)



        x_input_ids = []
        x_attention_masks = []
        y = []
        for d in data:
            input_txt = d.current_utterance
            if left_context:
                input_txt = f"{d.previous_utterance}{sep_text}" + input_txt
            if right_context:
                input_txt = input_txt + f"{sep_text}{d.next_utterance}"
            input_txt = input_txt + last_token

            x = tokenizer(input_txt, padding="max_length", truncation=True, max_length=input_length, return_tensors="pt")
            x_input_ids.append(x.input_ids[0])
            x_attention_masks.append(x.attention_mask[0])
            if d.gold_label == 1:
                y.append(torch.tensor([1], dtype=torch.float32))
            else:
                y.append(torch.tensor([0], dtype=torch.float32))

        super().__init__(x_input_ids=x_input_ids, x_attention_masks=x_attention_masks, y=y, tokenizerid=tokenizerid,
                         tokenizer=tokenizer)
