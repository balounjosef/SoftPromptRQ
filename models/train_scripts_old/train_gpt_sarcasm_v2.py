"""
@author White Wolf
"""

import os, sys
import shutil

# if we run this on Pilsen machine remotely
# sys.path.append("/mnt/data/PycharmProjects/Nancy_2022_experiments")

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Adafactor

import matplotlib.pyplot as plt

from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from models.gpt import GPTSoftPrompting
from models.utils.model_utils import loop_train, loop_test, prompts_norms, loop_train_with_f1, loop_test_with_f1

folds = 10
for fold in range(0, folds):
    print("EXPERIMENT WITH FOLD: ", fold)

    # basic settings
    modelidentifier = str(fold) + "_fold_gpt2_10_sarcasm_v2_fulltrain_adamW"
    gpt2id = "gpt2"
    device = None
    batch = 25
    epochs = 100
    learning_rate = 0.02
    save_after_ep = 50

    # Init folder
    modeldir = f"{__file__[:-len('models/train_scripts/train_gpt_sarcasm_v2.py')]}bin/{modelidentifier}_lr{learning_rate}_batch{batch}"
    if os.path.exists(modeldir):
        shutil.rmtree(modeldir)

    os.makedirs(modeldir, exist_ok=False)

    logfile = os.path.join(modeldir, "progress.csv")
    with open(logfile, "w", encoding="utf-8") as f:
        f.write("epoch\ttrain acc\ttrain loss\ttest acc\ttest loss\ttest f1\ttest p\ttest r\n")

    normsfile = os.path.join(modeldir, "norms.csv")

    shutil.copyfile(__file__, os.path.join(modeldir, "script.txt")) # copy this script for the details

    average_number_of_words = 40
    # Prepare data
    train_dataset = SarcasmV2Dataset(dataset_type="train", folds=10, fold_number=fold, tokenizerid=gpt2id, input_length=average_number_of_words)
    test_dataset = SarcasmV2Dataset(dataset_type="test", folds=10, fold_number=fold, tokenizerid=gpt2id, input_length=average_number_of_words)

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    # model loading
    model = GPTSoftPrompting(gpt2id, prompts=10, classes=1, device=None)
    print(f"{model} running on device {next(model.parameters()).device}")

    # training
    loss_fn = nn.BCEWithLogitsLoss()  # does not expect sigmoid
    # optimizer = Adafactor(model.parameters(), lr=learning_rate, weight_decay=1e-5, decay_rate=-0.8, scale_parameter=False, relative_step=False) # https://aclanthology.org/2021.emnlp-main.243.pdf
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    # first norms and init for plots
    minnorm, maxnorm, avgnorm, norms = prompts_norms(torch.cat([model.prompts, model.clsprompt]))
    with open(normsfile, "w", encoding="utf-8") as f:
        nstr = "\t".join([str(i) for i in range(1, len(norms))])
        f.write(f"epoch\t{nstr}\tCLS\n")
        nstr = "\t".join([str(n) for n in norms])
        f.write(f"0\t{nstr}\n")

    avg_norms = [avgnorm]
    min_norms = [minnorm]
    max_norms = [maxnorm]

    best_test_acc = 0.
    train_losses = [float("nan")]
    test_losses = [float("nan")]

    # train loop
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_acc, train_loss, train_f1, train_p, train_r = loop_train_with_f1(train_dataloader, model, loss_fn, optimizer)
        test_acc, test_loss, test_f1, test_p, test_r = loop_test_with_f1(val_dataloader, model, loss_fn)

        # save model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model, f'{modeldir}/model_{epoch}_train_{train_loss:>4f}_test_{test_loss:>4f}.pth')
        elif epoch % save_after_ep == 0:
            torch.save(model, f'{modeldir}/model_{epoch}_train_{train_loss:>4f}_test_{test_loss:>4f}.pth')

        # log progress
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{train_acc}\t{train_loss}\t{test_acc}\t{test_loss}\t{test_f1}\t{test_p}\t{test_r}\n")


        train_losses.append(train_loss)
        test_losses.append(test_loss)
        plt.figure()
        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.legend(["train", "test"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(f'{modeldir}/loss.pdf')
        plt.close()

        # log norms
        minnorm, maxnorm, avgnorm, norms = prompts_norms(torch.cat([model.prompts, model.clsprompt]))

        with open(normsfile, "a", encoding="utf-8") as f:
            nstr = "\t".join([str(n) for n in norms])
            f.write(f"{epoch}\t{nstr}\n")

        avg_norms.append(avgnorm)
        min_norms.append(minnorm)
        max_norms.append(maxnorm)
        plt.figure()
        plt.plot(avg_norms)
        plt.plot(min_norms)
        plt.plot(max_norms)
        plt.legend(["avg", "min", "max"])
        plt.xlabel("epoch")
        plt.ylabel("norm")
        plt.savefig(f'{modeldir}/norms.pdf')
        plt.close()

    del model
    torch.cuda.empty_cache()

    print("Done!")
