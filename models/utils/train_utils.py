import os
import torch
from sklearn import neighbors, metrics
from torch.utils.data import DataLoader

import paths
from models.utils.model_utils import prompts_norms
import numpy as np
import matplotlib.pyplot as plt


def loop_train_triplet(dataloader, model, loss_fn, optimizer):
    model.train(True)
    size = len(dataloader.dataset)
    total_loss, total_steps = 0.0, 0
    correct, total = 0, 0

    for batch, (anchor_sample,
                anchor_attention_mask,
                anchor_label,
                positive_sample,
                positive_attention_mask,
                negative_sample,
                negative_attention_mask) in enumerate(dataloader):

        # Compute prediction for anchor, positive, negative
        anchor_out = model(anchor_sample, anchor_attention_mask)
        positive_out = model(positive_sample, positive_attention_mask)
        negative_out = model(negative_sample, negative_attention_mask)

        # compute loss
        loss, distances_positive, distances_negative = loss_fn(anchor_out, positive_out, negative_out)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += torch.sum(distances_positive < distances_negative).cpu().detach().item()
        total += len(distances_positive)

        lossv = loss.item()
        total_loss += lossv
        total_steps += 1

        if batch % 10 == 0:
            current = batch * len(anchor_sample)
            print(f"[{current:>5d}/{size:>5d}] \t avg loss: {total_loss / total_steps:>7f} \t acc: {100 * (correct / total):>0.1f}%")

    mloss = total_loss / total_steps
    triplet_accuracy = correct / total

    print(f"Train triplet acc: {(100 * triplet_accuracy):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": triplet_accuracy,
        "loss": mloss
    }


def loop_test_triplet(dataloader, model, loss_fn):
    model.train(False)

    total_loss, total_steps = 0.0, len(dataloader)
    correct, total = 0, 0

    positives_outputs, negatives_outputs = [], []
    for anchor_sample,\
        anchor_attention_mask,\
        anchor_label,\
        positive_sample,\
        positive_attention_mask,\
        negative_sample,\
        negative_attention_mask in dataloader:

        # Compute prediction for anchor, positive, negative
        with torch.no_grad():
            anchor_out = model(anchor_sample, anchor_attention_mask)
            positive_out = model(positive_sample, positive_attention_mask)
            negative_out = model(negative_sample, negative_attention_mask)

        # compute loss
        loss, distances_positive, distances_negative = loss_fn(anchor_out, positive_out, negative_out)
        lossv = loss.item()
        total_loss += lossv

        correct += torch.sum(distances_positive < distances_negative).cpu().detach().item()
        total += len(distances_positive)


    mloss = total_loss / total_steps
    triplet_accuracy = correct / total

    print(f"Triplet acc: {(100 * triplet_accuracy):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": triplet_accuracy,
        "loss": mloss
    }


def loop_test_triplet_mean_nn_acc(dataloader, model, *args, **kwargs):
    model.train(False)
    resx = []
    resy = []
    with torch.no_grad():
        for x, xm, y in dataloader:
            preds = model(x, xm)
            resx.extend(preds.detach().cpu().tolist())
            resy.extend(y.flatten().long().detach().cpu().tolist())

    x = np.asarray(resx)
    y = np.asarray(resy)
    x0 = x[y == 0]
    x1 = x[y == 1]

    trainy = np.asarray([0, 1])

    # euc acc is straightforward
    trainx = np.asarray([np.mean(x0, axis=0), np.mean(x1, axis=0)])

    knn = neighbors.KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn.fit(trainx, trainy)
    predy = knn.predict(x)
    eucacc = metrics.accuracy_score(y, predy)

    # cos: 1) norm, 2) mean
    x0norm = x0 / np.linalg.norm(x0, axis=0)
    x1norm = x1 / np.linalg.norm(x1, axis=0)
    trainx = np.asarray([np.mean(x0norm, axis=0), np.mean(x1norm, axis=0)])

    knn = neighbors.KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(trainx, trainy)
    predy = knn.predict(x)
    cosacc = metrics.accuracy_score(y, predy)

    maxacc = max(eucacc, cosacc)
    print(f"1-NN test: acc={maxacc}, cosacc={cosacc}, eucacc={eucacc}")

    return {
        "acc": maxacc,
        "eucacc": eucacc,
        "cosacc": cosacc
    }


def loop_train_with_f1(dataloader, model, loss_fn, optimizer):
    model.train(True)
    size = len(dataloader.dataset)
    total_loss, total_steps = 0.0, 0
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0

    for batch, (x, xm, y) in enumerate(dataloader):
        y = y.to(model.device)

        # Compute prediction and loss
        pred = model(x, xm)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predt = pred >= 0  # no sigmoid


        # for computing tp, fp, fn
        pyand = predt * y
        pyand_inv = 1 - pyand

        tp += torch.sum(pyand).item()
        fp += torch.sum(pyand_inv * predt).item()
        fn += torch.sum(pyand_inv * y).item()

        correct += (predt == y).sum().item()
        total += torch.numel(y)

        lossv = loss.item()
        total_loss += lossv
        total_steps += 1

        if batch % 10 == 0:
            current = batch * len(x)
            print(f"[{current:>5d}/{size:>5d}] \t avg loss: {total_loss/total_steps:>7f} \t acc: {100*(correct/total):>0.1f}%")

    if tp > 0:
        f1 = tp / (tp + 0.5 * (fp + fn))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
    else:
        f1 = 0
        p = 0
        r = 0

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Train acc: {(100*acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    print(f"Train \n F1: {(100 * f1):>0.1f}%, P: {(100 * p):>0.1f}%, R: {(100 * r):>0.1f}% \n")
    return {
        "acc": acc,
        "loss": mloss,
        "f1": f1,
        "precision" : p,
        "recall" : r
    }

def loop_test_with_f1(dataloader, model, loss_fn):
    model.train(False)

    total_steps = len(dataloader)
    total_loss = 0.0
    correct, total = 0, 0

    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for x, xm, y in dataloader:
            y = y.to(model.device)

            pred = model(x, xm)     #[3,26]

            predt = pred >= 0  # no sigmoid

            # for computing tp, fp, fn
            pyand = predt * y
            pyand_inv = 1 - pyand

            tp += torch.sum(pyand).item()
            fp += torch.sum(pyand_inv * predt).item()
            fn += torch.sum(pyand_inv * y).item()

            correct += (predt == y).sum().item()
            total += torch.numel(y)

            lossv = loss_fn(pred, y).item()
            total_loss += lossv


    if tp > 0:
        f1 = tp / (tp + 0.5 * (fp + fn))
        p = tp / (tp + fp)
        r = tp / (tp + fn)
    else:
        f1 = 0
        p = 0
        r = 0

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Test acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    print(f"Test \n F1: {(100 * f1):>0.1f}%, P: {(100 * p):>0.1f}%, R: {(100 * r):>0.1f}% \n")
    return {
        "acc": acc,
        "loss": mloss,
        "f1": f1,
        "precision": p,
        "recall": r
    }


def loop_train_multiclass(dataloader, model, loss_fn, optimizer):
    model.train(True)
    size = len(dataloader.dataset)
    total_loss, total_steps = 0.0, 0
    correct, total = 0, 0

    for batch, (x, xm, y) in enumerate(dataloader):
        y = y.to(model.device)

        # Compute prediction and loss
        pred = model(x, xm)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # predt = pred >= 0  # no sigmoid
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += torch.numel(y)

        lossv = loss.item()
        total_loss += lossv
        total_steps += 1

        if batch % 10 == 0:
            current = batch * len(x)
            print(
                f"[{current:>5d}/{size:>5d}] \t avg loss: {total_loss / total_steps:>7f} \t acc: {100 * (correct / total):>0.1f}%")

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Train acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": acc,
        "loss": mloss
    }


def loop_test_multiclass(dataloader, model, loss_fn):
    model.train(False)

    total_steps = len(dataloader)
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for x, xm, y in dataloader:
            y = y.to(model.device)

            pred = model(x, xm)     #[3,26]

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += torch.numel(y)

            lossv = loss_fn(pred, y).item()
            total_loss += lossv

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Val acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": acc,
        "loss": mloss
    }

def loop_train(dataloader, model, loss_fn, optimizer):
    model.train(True)
    size = len(dataloader.dataset)
    total_loss, total_steps = 0.0, 0
    correct, total = 0, 0

    for batch, (x, xm, y) in enumerate(dataloader):
        y = y.to(model.device)

        # Compute prediction and loss
        pred = model(x, xm)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predt = pred >= 0  # no sigmoid
        correct += (predt == y).sum().item()
        total += torch.numel(y)

        lossv = loss.item()
        total_loss += lossv
        total_steps += 1

        if batch % 10 == 0:
            current = batch * len(x)
            print(f"[{current:>5d}/{size:>5d}] \t avg loss: {total_loss/total_steps:>7f} \t acc: {100*(correct/total):>0.1f}%")

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Train acc: {(100*acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": acc,
        "loss": mloss
    }


def loop_test(dataloader, model, loss_fn):
    model.train(False)

    total_steps = len(dataloader)
    total_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for x, xm, y in dataloader:
            y = y.to(model.device)

            pred = model(x, xm)     #[3,26]

            predt = pred >= 0  # no sigmoid
            correct += (predt == y).sum().item()
            total += torch.numel(y)

            lossv = loss_fn(pred, y).item()
            total_loss += lossv

    mloss = total_loss / total_steps
    acc = correct / total

    print(f"Val acc: {(100 * acc):>0.1f}%, avg loss: {mloss:>8f} \n")
    return {
        "acc": acc,
        "loss": mloss
    }


def _save_progress_plot(plot_res, key, path):
    legend = []
    plt.figure()
    for k, v in plot_res[key].items():
        plt.plot(v)
        legend.append(k)
    plt.legend(legend)
    plt.xlabel("epoch")
    plt.ylabel(key)
    plt.savefig(path)
    plt.close()
    

def _save_dicts_to_csv(ds, path):
    header = []
    values = []
    for d in ds:
        for k, v in d.items():
            header.append(k)
            values.append(str(v))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(values) + "\n")


def _log_norms(logstats, model, modeldir):
    pnorms = prompts_norms(model.prompts)
    logstats["prompt_norms"]["avg"].append(np.mean(pnorms))
    logstats["prompt_norms"]["min"].append(np.min(pnorms))
    logstats["prompt_norms"]["max"].append(np.max(pnorms))

    if model.clsprompt is not None:
        clsnorm = prompts_norms(model.clsprompt)[0]
        logstats["cls_norms"].append(clsnorm)

    plt.figure()
    plt.plot(logstats["prompt_norms"]["avg"])
    plt.plot(logstats["prompt_norms"]["min"])
    plt.plot(logstats["prompt_norms"]["max"])
    if model.clsprompt is None:
        plt.legend(["avg", "min", "max"])
    else:
        plt.plot(logstats["cls_norms"])
        plt.legend(["avg", "min", "max", "cls"])
    plt.xlabel("epoch")
    plt.ylabel("l2 norm")
    plt.savefig(f'{modeldir}/plot_norms.pdf')
    plt.close()

    if model.prompts_std is not None:
        pstdnorm = prompts_norms(model.prompts_std)
        logstats["prompt_std_norms"]["avg"].append(np.mean(pstdnorm))
        logstats["prompt_std_norms"]["min"].append(np.min(pstdnorm))
        logstats["prompt_std_norms"]["max"].append(np.max(pstdnorm))

        plt.figure()
        plt.plot(logstats["prompt_std_norms"]["avg"])
        plt.plot(logstats["prompt_std_norms"]["min"])
        plt.plot(logstats["prompt_std_norms"]["max"])
        plt.legend(["avg", "min", "max"])
        plt.xlabel("epoch")
        plt.ylabel("l2 norm")
        plt.savefig(f'{modeldir}/plot_stdnorms.pdf')
        plt.close()


def train_model(modelidentifier, model, train_dataset, val_dataset, batch, learning_rate=0.002, epochs=1000,
                train_loop_func=loop_train, val_loop_func=loop_test, save_obj="acc", save_after_each_ep=10,
                loss_fn=torch.nn.BCEWithLogitsLoss(), optimizer=None):
    """
    Parameters
    ----------
    modelidentifier (str) - will be used to create dir
    train_loop_func - func to train (follow loop_train params)
    val_loop_func - func to eval (follow loop_test params)
    save_obj - key returned by val_loop_func used to save best model - maximal value
    save_after_each_ep - model will be saved after specified num of epochs
    loss_fn - optional loss func
    optimizer - optional optimizer (if used learning_rate is ignored)
    """
    # print info
    print(f"{model.__class__.__name__} running on device {next(model.parameters()).device}")

    traininfo = {
        "modelidentifier": modelidentifier,
        "dataset": train_dataset.__class__.__name__,
        "tokenizerid": train_dataset.tokenizer_id,
        "input seq. length": train_dataset.get_input_len(),
        "model": model.__class__.__name__,
        "modelid": model.modelid,
        "base_model": model.model.__class__.__name__,
        "embeddim": model.embed_dim,
        "learnable params": model.get_num_of_learnable_params(),
        "prompts": model.numprompts,
        "variational_prompts": model.variational_prompts,
        "prompt_embed_text_init": model.prompt_embed_text_init,
        "classes": model.numclasses,
        "own_cls_head": model.own_cls_head,
        "epoch of best result": None,
        "batch": batch,
        "lr": learning_rate if optimizer is None else optimizer
    }

    print(traininfo)

    # Init folder
    modeldir = f"{paths.bin}{train_dataset.__class__.__name__}/{modelidentifier}_prompts{model.numprompts}_inp{train_dataset.get_input_len()}_lr{learning_rate}_batch{batch}"
    if os.path.exists(modeldir):
        print(f"'{modeldir}' exists")
        modeldir = modeldir + "_run"
        i = 2
        while os.path.exists(modeldir + str(i)):
            i += 1
        modeldir = modeldir + str(i)

    print(f"Destination dir decided: '{modeldir}'")
    os.makedirs(modeldir, exist_ok=False)
    progressfile = os.path.join(modeldir, "progress.csv")
    bestmodelcsv = os.path.join(modeldir, f"model_best_val_{save_obj}.csv")
    bestmodelfile = os.path.join(modeldir, f"model_best_val_{save_obj}.cp")

    # Prepare data
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # training
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # init for logs
    pnorms = prompts_norms(model.prompts)

    logstats = {
        "prompt_norms": {
            "avg": [np.mean(pnorms)],
            "min": [np.min(pnorms)],
            "max": [np.max(pnorms)],
        },
    }

    if model.clsprompt is not None:
        clsnorm = prompts_norms(model.clsprompt)[0]
        logstats["cls_norms"] = [clsnorm]

    if model.prompts_std is not None:
        pstdnorm = prompts_norms(model.prompts_std)
        logstats["prompt_std_norms"] = {
            "avg": [np.mean(pstdnorm)],
            "min": [np.min(pstdnorm)],
            "max": [np.max(pstdnorm)],
        }

    # progress log
    train_keys = [],
    val_keys = [],
    all_keys = [],
    plot_res = {}

    # save inited weights
    model.save_learnable_params(f'{modeldir}/model_0.cp')
    # train loop
    best_res = 0.
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_res = train_loop_func(train_dataloader, model, loss_fn, optimizer)
        val_res = val_loop_func(val_dataloader, model, loss_fn)

        # save model
        if epoch % save_after_each_ep == 0:
            model.save_learnable_params(f'{modeldir}/model_{epoch}.cp')
        if val_res[save_obj] > best_res:
            best_res = val_res[save_obj]
            model.save_learnable_params(bestmodelfile)

            traininfo["epoch of best result"] = epoch
            _save_dicts_to_csv([traininfo, val_res], bestmodelcsv)

        # log progress
        if epoch == 1:  # prepare
            train_keys = list(train_res.keys())
            val_keys = list(val_res.keys())
            all_keys = list(set(train_keys + val_keys))
            for k in all_keys:
                plot_res[k] = {
                    "train": [float("nan")],
                    "val": [float("nan")]
                }
            with open(progressfile, "a", encoding="utf-8") as f:
                f.write("\t".join(["epoch"] + ["train_" + k for k in train_keys] + ["val_" + k for k in val_keys]) + "\n")

        # log results
        with open(progressfile, "a", encoding="utf-8") as f:
            tmp = [epoch] + [train_res[k] for k in train_keys] + [val_res[k] for k in val_keys]
            f.write("\t".join([str(k) for k in tmp]) + "\n")

        for k in train_keys:
            plot_res[k]["train"].append(train_res[k])

        for k in val_keys:
            plot_res[k]["val"].append(val_res[k])

        for k in all_keys:
            _save_progress_plot(plot_res, key=k, path=f'{modeldir}/plot_{k}.pdf')

        # log norms
        _log_norms(logstats, model, modeldir)

    print("Done!")


if __name__ == '__main__':
    from dataset.rquet_dataset import RquetDataset
    from models.GPT2SoftPrompt import GPT2SoftPromptModel

    modelid = "gpt2"
    train_model(
        "test",
        model=GPT2SoftPromptModel(modelid, prompts=2),
        train_dataset=RquetDataset("train", modelid, input_length=29),
        val_dataset=RquetDataset("test", modelid, input_length=29),
        batch=4,
        learning_rate=0.002
    )
    # ds = RquetDataset("test", modelid, input_length=29)
    # ds.y = [ds.y[i] for i in range(8)]
    # train_model(
    #     "test",
    #     model=GPT2SoftPromptModel(modelid, prompts=2),
    #     train_dataset=ds,
    #     val_dataset=ds,
    #     batch=4,
    #     learning_rate=0.002
    # )
