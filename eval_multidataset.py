from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import paths
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.utils.train_utils import loop_test_with_f1
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
import os


inputlen = 128-10

# TODO set weights - ideally ordered accordingly to table
model_weights_files = [
    paths.bin + "SwDA_RQDataset_Balanced/bert-base-cased_PREV_Q_NEXT_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "SwDA_RQDataset_Balanced/bert-base-cased_PREV_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "SwDA_RQDataset_Balanced/bert-base-cased_Q_NEXT_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "SwDA_RQDataset_Balanced/bert-base-cased_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    # ######
    paths.bin + "RquetDataset/bert-base-cased_PREV_Q_NEXT_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "RquetDataset/bert-base-cased_PREV_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "RquetDataset/bert-base-cased_Q_NEXT_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "RquetDataset/bert-base-cased_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    # ######
    paths.bin + "SarcasmV2Dataset/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    # ######
    paths.bin + "MRDADataset/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    paths.bin + "MRDADataset_Balanced/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp"
]

for mwf in model_weights_files:
    assert os.path.exists(mwf), f"File '{mwf}' not found"

# TODO select model
# modelid = "gpt2"
# model = GPT2SoftPromptTripletCLSModel(modelid, prompts=10, out_dim=1)
modelid = "bert-base-cased"
model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=1)



## Body of the script
# Accuracy and F1, P, R for positive class
csv_files = {
    "acc": "bin/eval_multidataset_acc.csv",
    "f1": "bin/eval_multidataset_f1.csv",
    "precision": "bin/eval_multidataset_p.csv",
    "recall": "bin/eval_multidataset_r.csv",
}
# Datasets ordered accordingly to table
datasets = {
    "SwDA_PQN": SwDA_RQDataset_Balanced("test", modelid, input_length=inputlen, left_context=True, right_context=True),
    "SwDA_PQ": SwDA_RQDataset_Balanced("test", modelid, input_length=inputlen, left_context=True, right_context=False),
    "SwDA_QN": SwDA_RQDataset_Balanced("test", modelid, input_length=inputlen, left_context=False, right_context=True),
    "SwDA_Q": SwDA_RQDataset_Balanced("test", modelid, input_length=inputlen, left_context=False, right_context=False),
    "Rquet_PQN": RquetDataset("test", modelid, input_length=inputlen, left_context=True, right_context=True),
    "Rquet_PQ": RquetDataset("test", modelid, input_length=inputlen, left_context=True, right_context=False),
    "Rquet_QN": RquetDataset("test", modelid, input_length=inputlen, left_context=False, right_context=True),
    "Rquet_Q": RquetDataset("test", modelid, input_length=inputlen, left_context=False, right_context=False),
    "SarcasmV2": SarcasmV2Dataset("test", modelid, input_length=inputlen),
    "MRDA": MRDADataset("test", modelid, input_length=inputlen),
    "MRDA_Balanced": MRDADataset_Balanced("test", modelid, input_length=inputlen)
}

dskeys = datasets.keys()
header = "\t".join(["model"] + list(dskeys))

for k, v in csv_files.items():
    with open(v, "w", encoding="utf-8") as f:
        f.write(header)

for mwf in model_weights_files:
    print(f"Processing of '{mwf}' started:")
    # write model info
    for k, v in csv_files.items():
        with open(v, "a", encoding="utf-8") as f:
            f.write("\n" + mwf)

    model.load_learnable_params(mwf)

    for dsk in dskeys:
        ds = datasets[dsk]
        res = loop_test_with_f1(dataloader=DataLoader(ds, batch_size=64, shuffle=False), model=model, loss_fn=BCEWithLogitsLoss())

        # write result for given dataset and metric (k)
        for k, v in csv_files.items():
            with open(v, "a", encoding="utf-8") as f:
                f.write(f"\t{res[k]}")

        print(f"\t {mwf} \t {dsk} \t acc: {res['acc']}")
