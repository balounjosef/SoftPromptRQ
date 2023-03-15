from dataset.triplet_dataset import TripletMultiDataset, TripletMultiDatasetVal
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.utils.model_utils import freeze_model
from models.utils.train_utils import train_model, loop_train_triplet, loop_test_triplet
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
from triplet_loss import TripletLoss



# contrastively train the model Step 1

inputlen = 128-10   # always the same

outdim = 32

modelid = "bert-base-cased"



model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=outdim)

p = "train"
train_ds = TripletMultiDataset(
    [
        MRDADataset(p, modelid, input_length=inputlen),
        RquetDataset(p, modelid, input_length=inputlen, left_context=False, right_context=False),
    ]
)
p = "val"
val_ds = TripletMultiDatasetVal(
    [
        MRDADataset_Balanced(p, modelid, input_length=inputlen),
        RquetDataset(p, modelid, input_length=inputlen, left_context=False, right_context=False),
    ]
)


margin=0.5

train_model(
    f"{modelid}_out{outdim}_margin{margin}_mrda_rquet",
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch=32,
    learning_rate=0.0002,
    epochs=250,
    loss_fn=TripletLoss(margin=margin),
    train_loop_func=loop_train_triplet,
    val_loop_func=loop_test_triplet,
    save_obj="acc"
)
