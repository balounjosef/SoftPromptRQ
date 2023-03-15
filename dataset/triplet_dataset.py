import random
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, anchor_ds, sample_ds=None, inverse_sample_label=False, random_seed=None):
        super().__init__()
        if sample_ds is None:
            sample_ds = anchor_ds
            self.same_ds = True
            self.tokenizer_id = anchor_ds.tokenizer_id
        else:
            self.same_ds = False
            self.tokenizer_id = f"{anchor_ds.tokenizer_id}={anchor_ds.__class__.__name__}({sample_ds.tokenizer_id}={sample_ds.__class__.__name__})"

        self.random = random.Random(random_seed)

        self.anchor_pos = []
        self.anchor_neg = []
        for x, xm, y in anchor_ds:
            if y[0].item() > 0.5:
                self.anchor_pos.append([x, xm, y])
            else:
                self.anchor_neg.append([x, xm, y])

        self.sample_pos = []
        self.sample_neg = []
        for x, xm, y in sample_ds:
            if (y[0].item() > 0.5) != inverse_sample_label:
                self.sample_pos.append([x, xm, y])
            else:
                self.sample_neg.append([x, xm, y])

    def get_input_len(self):
        if self.same_ds:
            return len(self.anchor_pos[0][0])
        return f"{len(self.anchor_pos[0][0])}({len(self.sample_pos[0][0])})"

    def __len__(self):
        return len(self.anchor_pos) + len(self.anchor_neg)

    def __getitem__(self, idx):
        if idx < len(self.anchor_pos):
            anchor_id = idx
            anchor = self.anchor_pos[anchor_id]
            pos = self.sample_pos
            neg = self.sample_neg
        else:
            anchor_id = idx - len(self.anchor_pos)
            anchor = self.anchor_neg[anchor_id]
            pos = self.sample_neg
            neg = self.sample_pos

        pos_random_index = self.random.randrange(len(pos))
        if self.same_ds:
            while pos_random_index == anchor_id:
                pos_random_index = self.random.randrange(len(pos))

        neg_random_index = self.random.randrange(len(neg))

        positive = pos[pos_random_index]
        negative = neg[neg_random_index]

        # anchor_sample, anchor_attention_mask, anchor_label, positive_sample, positive_attention_mask, negative_sample, negative_attention_mask
        return anchor[0], anchor[1], anchor[2], positive[0], positive[1], negative[0], negative[1]


class TripletDatasetVal(Dataset):
    """
    Always the same triplets
    """
    def __init__(self, anchor_ds, sample_ds=None, inverse_sample_label=False, random_seed=7):
        super().__init__()
        ds = TripletDataset(anchor_ds=anchor_ds, sample_ds=sample_ds, inverse_sample_label=inverse_sample_label, random_seed=random_seed)

        self.tokenizer_id = ds.tokenizer_id
        self.input_len = ds.get_input_len()

        self.data = []
        for d in ds:   # prepare constant triplets
            self.data.append(d)

    def get_input_len(self):
        return self.input_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TripletMultiDataset(Dataset):
    """
    Random triplets
    """
    def __init__(self, datasets: list, random_seed=None, shots_per_dataset=None):
        super().__init__()
        self.tokenizer_id = datasets[0].tokenizer_id
        self.input_len = datasets[0].get_input_len()
        for ds in datasets:
            assert ds.tokenizer_id == self.tokenizer_id
            assert ds.get_input_len() == self.input_len
        self.tokenizer_id += "(" + ", ".join([ds.__class__.__name__ for ds in datasets]) + ")"

        triplet_datasets = []
        for ds in datasets:
            if shots_per_dataset is not None:
                ds.few_shot(shots_per_dataset)
            triplet_datasets.append(TripletDataset(ds, random_seed=random_seed))

        lookuptbl = []
        for tds in triplet_datasets:
            for i in range(len(tds)):
                lookuptbl.append([tds, i])

        self.lookuptbl = lookuptbl

    def get_input_len(self):
        return self.input_len

    def __len__(self):
        return len(self.lookuptbl)

    def __getitem__(self, idx):
        tds, i = self.lookuptbl[idx]
        return tds[i]


class TripletMultiDatasetVal(TripletMultiDataset):
    """
    Always the same triplets
    """
    def __init__(self, datasets: list, random_seed=7):
        super().__init__(datasets, random_seed)
        self.data = []
        for tds, i in self.lookuptbl:   # prepare constant triplets
            self.data.append(tds[i])
        self.lookuptbl = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    from dataset.rquet_dataset import RquetDataset
    ds = RquetDataset("val", "bert-base-cased", input_length=40)
    dst = TripletDataset(anchor_ds=ds)
    dsv = TripletDatasetVal(anchor_ds=ds)

    for x in ds:
        print(x)