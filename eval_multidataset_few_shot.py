from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import paths
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.utils.model_utils import predict_ds
from models.utils.train_utils import loop_test_with_f1
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
import os


shots_per_class = 1
n_neighbors = 1
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
    paths.bin + "MRDADataset_Balanced/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp",
    # ######
    None
]
for mwf in model_weights_files:
    if mwf is not None:
        assert os.path.exists(mwf), f"File '{mwf}' not found"

# TODO select model
# modelid = "gpt2"
# model = GPT2SoftPromptTripletCLSModel(modelid, prompts=10, out_dim=None)
modelid = "bert-base-cased"
model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=None)

## Body of the script


# Accuracy and F1, P, R for positive class
csv_files = {
    "acc": f"eval_multidataset_fewshot{shots_per_class}_acc.csv",
    # "f1": "eval_multidataset_f1.csv",
    # "precision": "eval_multidataset_p.csv",
    # "recall": "eval_multidataset_r.csv",
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

# val Datasets ordered accordingly to table
valdatasets = {
    "SwDA_PQN": SwDA_RQDataset_Balanced("val", modelid, input_length=inputlen, left_context=True, right_context=True),
    "SwDA_PQ": SwDA_RQDataset_Balanced("val", modelid, input_length=inputlen, left_context=True, right_context=False),
    "SwDA_QN": SwDA_RQDataset_Balanced("val", modelid, input_length=inputlen, left_context=False, right_context=True),
    "SwDA_Q": SwDA_RQDataset_Balanced("val", modelid, input_length=inputlen, left_context=False, right_context=False),
    "Rquet_PQN": RquetDataset("val", modelid, input_length=inputlen, left_context=True, right_context=True),
    "Rquet_PQ": RquetDataset("val", modelid, input_length=inputlen, left_context=True, right_context=False),
    "Rquet_QN": RquetDataset("val", modelid, input_length=inputlen, left_context=False, right_context=True),
    "Rquet_Q": RquetDataset("val", modelid, input_length=inputlen, left_context=False, right_context=False),
    "SarcasmV2": SarcasmV2Dataset("val", modelid, input_length=inputlen),
    "MRDA": MRDADataset("val", modelid, input_length=inputlen),
    "MRDA_Balanced": MRDADataset_Balanced("val", modelid, input_length=inputlen)
}

for k, v in valdatasets.items():
    v.few_shot(shots_per_class=shots_per_class)

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
            f.write(f"\n{mwf}")

    if mwf is None:
        basemodel = model.model

        if modelid == "gpt2":
            model = lambda x, xmask: basemodel(x.to(basemodel.device), xmask.to(basemodel.device)).last_hidden_state[:, -1]
        elif modelid == "bert-base-cased":
            model = lambda x, xmask: basemodel(x.to(basemodel.device), xmask.to(basemodel.device)).last_hidden_state[:, 0]
        else:
            raise NotImplementedError()
    else:
        model.load_learnable_params(mwf)

    for dsk in dskeys:
        ds = datasets[dsk]
        valds = valdatasets[dsk]

        valx, valy = predict_ds(valds, model)
        testx, testy = predict_ds(ds, model)

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(valx, valy)
        predy = knn.predict(testx)

        acc = accuracy_score(testy, predy)

        with open(csv_files["acc"], "a", encoding="utf-8") as f:
            f.write(f"\t{acc}")

        print(f"\t {mwf} \t {dsk} \t acc: {acc}")
