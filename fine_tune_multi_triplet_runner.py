from dataset.combined_dataset import CombinedDataset
from dataset.triplet_dataset import TripletMultiDataset, TripletMultiDatasetVal, TripletDataset
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.GPT2SoftPrompt import GPT2SoftPromptTripletCLSModel
from models.utils.train_utils import train_model, loop_train_triplet, loop_test_triplet, loop_test_triplet_mean_nn_acc
from dataset.mrda_dataset import MRDADataset, MRDADataset_Balanced
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
from triplet_loss import TripletLoss
import sys


# contrastively train the model Step 2

def run_experiment(shots, seed, epochs):
    inputlen = 128-10   # always the same

    outdim = 32

    modelid = "bert-base-cased"
    model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=outdim)

    modelpath = "bin/TripletMultiDataset/bert-base-cased_out32_margin0.5_rquet_mrda_prompts10_inp118_lr0.0002_batch32/model_best_val_acc.cp"
    model.load_learnable_params(modelpath)

    combined_swda_mrda_rquet_ds = CombinedDataset([
            SwDA_RQDataset_Balanced("train", modelid, input_length=inputlen).few_shot(shots, seed=seed),
            RquetDataset("train", modelid, input_length=inputlen),
            MRDADataset("train", modelid, input_length=inputlen)
        ])

    train_dataset = TripletDataset(
        anchor_ds=combined_swda_mrda_rquet_ds,
        sample_ds=SwDA_RQDataset_Balanced("train", modelid, input_length=inputlen).few_shot(shots, seed=seed)
    )

    margin=0.5

    train_model(
        f"{modelid}_out{outdim}_margin{margin}_fine_tuning_swda_shots{shots}_seed{seed}",
        model=model,
        train_dataset=train_dataset,
        val_dataset=train_dataset,
        batch=32,
        learning_rate=0.0002,
        epochs=epochs,
        loss_fn=TripletLoss(margin=margin),
        train_loop_func=loop_train_triplet,
        val_loop_func=loop_test_triplet,
        save_obj="acc"
    )


if __name__ == '__main__':
    shot = int(sys.argv[1])
    seed = int(sys.argv[2])
    epochs = int(sys.argv[3])
    run_experiment(shot, seed, epochs)
    #print("python fine_tune_multi_triplet_runner.py " + str(shot) + " " + str(seed))