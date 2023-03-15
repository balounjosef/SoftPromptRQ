from dataset.abstract_ds import AbstractDataset


class CombinedDataset(AbstractDataset):
    def __init__(self, datasets):
        tokenizerid = datasets[0].tokenizer_id

        x_input_ids = []
        x_attention_masks = []
        y = []

        for ds in datasets:
            assert isinstance(ds, AbstractDataset)
            assert ds.tokenizer_id == tokenizerid

            x_input_ids.extend(ds.x_input_ids)
            x_attention_masks.extend(ds.x_attention_masks)
            y.extend(ds.y)

        super().__init__(x_input_ids=x_input_ids, x_attention_masks=x_attention_masks, y=y, tokenizerid=tokenizerid, tokenizer=tokenizerid)


if __name__ == '__main__':
    from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
    from dataset.rquet_dataset import RquetDataset

    ds = CombinedDataset([
        SwDA_RQDataset_Balanced("test", "gpt2", input_length=10),
        RquetDataset("test", "gpt2", input_length=10)
    ])

    print(len(ds))
    print(ds)