from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import paths
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.T5SoftPrompt import T5SoftPromptModel
from models.utils.train_utils import loop_test_with_f1, loop_test
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
import os


inputlen = 128-10

prompts = 10

modelid = "google/flan-t5-large"
model = T5SoftPromptModel(modelid, prompts=prompts, out_dim=1)


# dataset = RquetDataset("test", modelid, input_length=inputlen, left_context=False, right_context=False)
# dataset = MRDADataset_Balanced("test", modelid, input_length=inputlen)
dataset = SwDA_RQDataset_Balanced("test", modelid, input_length=inputlen)


fname = "model_best_val_acc"


ds_id = dataset.__class__.__name__

modelwp = paths.bin + f"{ds_id}/{modelid}_{prompts}_prompts{prompts}_inp{inputlen}_lr0.0002_batch24/{fname}.cp"

model.load_learnable_params(modelwp)

res = loop_test(dataloader=DataLoader(dataset, batch_size=24, shuffle=False), model=model, loss_fn=BCEWithLogitsLoss())

print(res)
