import sys

import paths
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.utils.train_utils import train_model
from dataset.mrda_dataset import MRDADataset_Balanced
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced


def run_experiment(shots_per_class, batch, epochs, ds, seed=7, transfer=None):
    assert batch <= shots_per_class
    assert ds in ["mrda", "swda"]
    assert transfer in ["mrda", "swda", "rquet", "3corp", None]

    inputlen = 128-10   # always the same

    modelid = "bert-base-cased"
    model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=1)
    if transfer is not None:
        if transfer == "swda":
            model.load_learnable_params(paths.bin + "SwDA_RQDataset_Balanced/bert-base-cased_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")
        elif transfer == "rquet":
            model.load_learnable_params(paths.bin + "RquetDataset/bert-base-cased_Q_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")
        elif transfer == "mrda":
            model.load_learnable_params(paths.bin + "MRDADataset_Balanced/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")
        elif transfer == "3corp":
            model.load_learnable_params(paths.bin + "CombinedDataset/bert-base-cased_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")


    modelidentifier = modelid
    if ds == "mrda":
        train_ds = MRDADataset_Balanced("train", modelid, input_length=inputlen)
        val_ds = MRDADataset_Balanced("val", modelid, input_length=inputlen)
    else:
        train_ds = SwDA_RQDataset_Balanced("train", modelid, input_length=inputlen, left_context=False, right_context=False)
        val_ds = SwDA_RQDataset_Balanced("val", modelid, input_length=inputlen, left_context=False, right_context=False)
        modelidentifier = f"{modelidentifier}_Q"

    train_ds.few_shot(shots_per_class=shots_per_class, seed=seed)
    modelidentifier = f"{modelidentifier}_{shots_per_class}shot_{seed}seed"
    if transfer is not None:
        modelidentifier = f"{modelidentifier}_transfer_{transfer}"

    train_model(
        modelidentifier=modelidentifier,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch=batch,
        learning_rate=0.0002,
        epochs=epochs,
        save_after_each_ep=1
    )

if __name__ == '__main__':
    transfer = None
    if len(sys.argv) > 6:
        transfer = sys.argv[6]

    run_experiment(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), transfer)