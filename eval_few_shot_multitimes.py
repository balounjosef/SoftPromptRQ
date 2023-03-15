import math
import os

import numpy as np
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

import paths
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.utils.train_utils import loop_test_with_f1


inputlen = 128-10
modelid = "bert-base-cased"

evalpath = paths.bin + "eval_multitime_mrdatransfer.csv"

# TODO select DS

with open(evalpath, "w", encoding="utf-8") as f:
    f.write("10 random prototype runs results **********************\n")

for dspart in ["test", "all"]:
    dataset = SwDA_RQDataset_Balanced(dspart, modelid, input_length=inputlen, left_context=False, right_context=False)
    with open(evalpath, "a", encoding="utf-8") as f:
        f.write(f"\n{dspart} ##################################\n")

    for fname in ["model_best_val_acc", "model_30"]:
        print(f"\n{fname} **********************")
        with open(evalpath, "a", encoding="utf-8") as f:
            f.write(f"\n{fname} **********************\n")


        # TODO set weights - ideally ordered accordingly to table
        model_weights_files = [
            [
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_0seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_1seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_2seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_3seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_4seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_5seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_6seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_7seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_8seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_1shot_9seed_transfer_prompts10_inp118_lr0.0002_batch1/{fname}.cp",
            ],
            [
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_0seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_1seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_2seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_3seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_4seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_5seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_6seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_7seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_8seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_5shot_9seed_transfer_prompts10_inp118_lr0.0002_batch4/{fname}.cp",
            ],
            [
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_0seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_1seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_2seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_3seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_4seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_5seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_6seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_7seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_8seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_10shot_9seed_transfer_prompts10_inp118_lr0.0002_batch8/{fname}.cp",
            ],
            [
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_0seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_1seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_2seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_3seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_4seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_5seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_6seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_7seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_8seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
                paths.bin + f"SwDA_RQDataset_Balanced/bert-base-cased_Q_25shot_9seed_transfer_prompts10_inp118_lr0.0002_batch16/{fname}.cp",
            ]
        ]

        for mwfl in model_weights_files:
            for mwf in mwfl:
                assert os.path.exists(mwf), f"File '{mwf}' not found"

        model = BertSoftPromptTripletCLSModel(modelid, prompts=10, out_dim=1)

        ## Body of the scripy
        for mwfl in model_weights_files:
            results = []
            for mwf in mwfl:
                print(f"processing {mwf}")
                model.load_learnable_params(mwf)

                res = loop_test_with_f1(dataloader=DataLoader(dataset, batch_size=64, shuffle=False), model=model, loss_fn=BCEWithLogitsLoss())
                results.append(res["acc"])

            meanacc = np.mean(results)
            stdacc = np.std(results)
            confint = 1.96 * stdacc / math.sqrt(len(results))

            print(f"{meanacc}\t+-\t{confint}")
            with open(evalpath, "a", encoding="utf-8") as f:
                f.write(f"{meanacc}\t+-\t{confint}\n")