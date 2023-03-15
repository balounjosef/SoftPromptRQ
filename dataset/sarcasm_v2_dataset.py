"""
@author White Wolf
"""
import torch
from dataset.abstract_ds import AbstractDataset
from models.utils.model_utils import decide_tokenizer
from sarcasm_v2_experiments import sarcasm_v2_data_loader
import random


def balance_length(x_input_ids, x_attention_masks, y, drop_longer=110, binsize=10):
    assert drop_longer % binsize == 0
    rnd = random.Random(7)

    bins0 = []
    bins1 = []
    for i in range(drop_longer//binsize):
        bins0.append([])
        bins1.append([])

    for i in range(len(x_input_ids)):
        xid = x_input_ids[i]
        xmask = x_attention_masks[i]
        xy = y[i]

        tokenslen = torch.sum(xmask).item()
        if tokenslen > 110:
            continue

        binid = (tokenslen - 1) // binsize    # 1 - 10 -> bin 0
        if xy > 0.5:
            bins1[binid].append((xid, xmask, xy))
        else:
            bins0[binid].append((xid, xmask, xy))

    resids = []
    resmasks = []
    resy = []
    for binid in range(len(bins0)):
        bin0 = bins0[binid]
        bin1 = bins1[binid]
        selectn = min(len(bin0), len(bin1))

        rnd.shuffle(bin0)
        rnd.shuffle(bin1)

        for seli in range(selectn):
            rid, rmask, ry = bin0[seli]
            resids.append(rid)
            resmasks.append(rmask)
            resy.append(ry)

            rid, rmask, ry = bin1[seli]
            resids.append(rid)
            resmasks.append(rmask)
            resy.append(ry)

    return resids, resmasks, resy


class SarcasmV2Dataset(AbstractDataset):
    def __init__(self, part, tokenizerid, input_length=128-10, folds=10, fold_number=0, balance_len=False):
        assert part in ["test", "val", "train"]
        dataset = "GEN"
        assert fold_number < folds, " fold number is above folds"
        sarcasm_v2_data, train_folds, test_folds = sarcasm_v2_data_loader.load_sarcasm_v2_data(dataset, folds=folds)
        # utilize disjunctive test folds: fold_number as test, next fold as val, other folds as train

        testfold = fold_number
        valfold = (testfold + 1) % folds
        if part == "test":
            data_ids = test_folds[testfold]
        elif part == "val":
            data_ids = test_folds[valfold]
        elif part == "train":
            data_ids = []
            for i in range(folds):
                if i != testfold and i != valfold:
                    data_ids.extend(test_folds[i])

        tokenizer, sep_text, last_token = decide_tokenizer(tokenizerid)

        x_input_ids = []
        x_attention_masks = []
        y = []
        for id in data_ids:
            d = sarcasm_v2_data[id]
            # <|endoftext|> used as a separator
            input_txt = f"{d.text}{last_token}"
            x = tokenizer(input_txt, padding="max_length", truncation=True, max_length=input_length, return_tensors="pt")
            x_input_ids.append(x.input_ids[0])
            x_attention_masks.append(x.attention_mask[0])
            if d.gold_label == "sarc":
                y.append(torch.tensor([1], dtype=torch.float32))
            else:
                y.append(torch.tensor([0], dtype=torch.float32))

        if balance_len:
            x_input_ids, x_attention_masks, y = balance_length(x_input_ids, x_attention_masks, y)

        super().__init__(x_input_ids=x_input_ids, x_attention_masks=x_attention_masks, y=y, tokenizerid=tokenizerid, tokenizer=tokenizer)


if __name__ == '__main__':
    ds = SarcasmV2Dataset("val", "bert-base-cased", input_length=118, balance_len=True)

    print(ds)