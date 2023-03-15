from dataset.combined_dataset import CombinedDataset
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.utils.train_utils import train_model
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced


inputlen = 128-10   # always the same

modelid = "bert-base-cased"
model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=1)


p = "train"
train_ds = CombinedDataset([
    MRDADataset_Balanced(p, modelid, input_length=inputlen),    # should be balanced
    RquetDataset(p, modelid, input_length=inputlen, left_context=False, right_context=False),
])

p = "val"
val_ds = CombinedDataset(
    [
        MRDADataset_Balanced(p, modelid, input_length=inputlen),
        RquetDataset(p, modelid, input_length=inputlen, left_context=False, right_context=False),
    ]
)

train_model(
    f"{modelid}",
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch=64,
    learning_rate=0.0002,
    epochs=300
)
