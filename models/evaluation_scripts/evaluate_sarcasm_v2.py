"""
@author White Wolf
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_average_results_with_stdev(folds_dictionary, metric):
    fold_metric_list = []
    folds_optimal_epochs = {}
    # find optimal epoch --> best f1 test score
    for fold, dataframe in folds_dictionary.items():
        f1_values = dataframe["test f1"].to_numpy()
        max_value, best_epoch = np.max(f1_values), np.argmax(f1_values)
        folds_optimal_epochs[fold] = best_epoch

    for fold, dataframe in folds_dictionary.items():
        optimal_epoch = folds_optimal_epochs[fold]
        value = dataframe[metric][optimal_epoch]
        fold_metric_list.append(value)

    # last epoch results
    fold_metrics = np.array(fold_metric_list)
    avg_metric = round(np.average(fold_metrics), 3)
    std_dev =    round(np.std(fold_metrics), 3)
    return avg_metric, std_dev, folds_optimal_epochs

def plot_f1_prec_rec_acc(folds_dictionary, folder_results):
    fold_f1_list, fold_prec_list, fold_rec_list, fold_acc_list = [], [], [], []

    for fold, dataframe in folds_dictionary.items():
        f1s = dataframe["test f1"].to_numpy()
        precs = dataframe["test p"].to_numpy()
        recs = dataframe["test r"].to_numpy()
        accs = dataframe["test acc"].to_numpy()

        fold_f1_list.append(f1s)
        fold_prec_list.append(precs)
        fold_rec_list.append(recs)
        fold_acc_list.append(accs)

    fold_f1s = np.array(fold_f1_list)
    fold_precs = np.array(fold_prec_list)
    fold_recs = np.array(fold_rec_list)
    fold_accs = np.array(fold_acc_list)

    folds_average_f1s = np.average(fold_f1s, axis=0)
    folds_average_precs = np.average(fold_precs, axis=0)
    folds_average_recs = np.average(fold_recs, axis=0)
    folds_average_accs = np.average(fold_accs, axis=0)

    fill_between_list_f1_up = folds_average_f1s + np.std(folds_average_f1s, axis=0)
    fill_between_list_f1_down = folds_average_f1s - np.std(folds_average_f1s, axis=0)
    fill_between_list_p_up = folds_average_precs + np.std(folds_average_precs, axis=0)
    fill_between_list_p_down = folds_average_precs - np.std(folds_average_precs, axis=0)
    fill_between_list_r_up = folds_average_recs + np.std(folds_average_recs, axis=0)
    fill_between_list_r_down = folds_average_recs - np.std(folds_average_recs, axis=0)
    fill_between_list_acc_up = folds_average_accs + np.std(folds_average_accs, axis=0)
    fill_between_list_acc_down = folds_average_accs - np.std(folds_average_accs, axis=0)

    epochs = np.arange(0, 100)

    plt.figure()
    plt.plot(epochs, folds_average_precs, color="brown", label="Average Precision")
    plt.fill_between(epochs, fill_between_list_p_up, fill_between_list_p_down, alpha=.1)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.savefig(folder_results+"precision.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs, folds_average_recs, color="red", label="Average Recall")
    plt.fill_between(epochs, fill_between_list_r_up, fill_between_list_r_down, alpha=.1)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.savefig(folder_results + "recall.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs, folds_average_f1s, color="black", label="Average F1")
    plt.fill_between(epochs, fill_between_list_f1_up, fill_between_list_f1_down, alpha=.1)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.savefig(folder_results + "f1.pdf")
    plt.close()

    plt.figure()
    plt.plot(epochs, folds_average_accs, color="green", label="Average Accuracy")
    plt.fill_between(epochs, fill_between_list_acc_up, fill_between_list_acc_down, alpha=.1)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(folder_results + "acc.pdf")
    plt.close()


def plot_train_test_loss(folds_dictionary, folder_results):


    fold_train_loss_list, fold_test_loss_list = [], []
    for fold, dataframe in folds_dictionary.items():
        train_losses = dataframe["train loss"].to_numpy()
        test_losses = dataframe["test loss"].to_numpy()
        fold_train_loss_list.append(train_losses)
        fold_test_loss_list.append(test_losses)

    fold_train_losses = np.array(fold_train_loss_list)
    fold_test_losses = np.array(fold_test_loss_list)

    folds_average_train_losses = np.average(fold_train_losses, axis=0)
    folds_average_test_losses = np.average(fold_test_losses, axis=0)

    fill_between_list_train_loss_up = folds_average_train_losses + np.std(fold_train_losses, axis=0)
    fill_between_list_train_loss_down = folds_average_train_losses - np.std(fold_train_losses, axis=0)
    fill_between_list_test_loss_up  = folds_average_test_losses + np.std(fold_test_losses, axis=0)
    fill_between_list_test_loss_down = folds_average_test_losses - np.std(fold_test_losses, axis=0)

    epochs = np.arange(0, 100)

    plt.figure()
    plt.plot(epochs, folds_average_train_losses, label="Average Train Loss")
    plt.fill_between(epochs, fill_between_list_train_loss_up, fill_between_list_train_loss_down, alpha=.1)
    plt.plot(epochs, folds_average_test_losses, label="Average Test Loss")
    plt.fill_between(epochs, fill_between_list_test_loss_up, fill_between_list_test_loss_down, alpha=.1)
    plt.ylim(0, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(folder_results+"train_test_loss.pdf")
    plt.close()

def evaluate_results(folder_results):
    folds_dictionary = {}
    for progress_file in os.listdir(folder_results):
        if progress_file.endswith(".csv"):
            folder_number = progress_file.split("_")[1]
            dataframe = pd.read_csv(os.path.join(folder_results, progress_file), sep='\t')
            folds_dictionary[folder_number] = dataframe

    # 1) plot train & test loss
    plot_train_test_loss(folds_dictionary, folder_results=folder_results)

    # 2) plot average f1, prec, rec, acc
    plot_f1_prec_rec_acc(folds_dictionary, folder_results=folder_results)

    # 3) get average f1, prec, rec, acc with std_dev (in the best epoch in each fold)
    avg_f1, f1_std_dev, folds_optimal_epochs = get_average_results_with_stdev(folds_dictionary, "test f1")
    avg_prec, prec_std_dev, folds_optimal_epochs = get_average_results_with_stdev(folds_dictionary, "test p")
    avg_rec, rec_std_dev, folds_optimal_epochs = get_average_results_with_stdev(folds_dictionary, "test r")
    avg_acc, acc_std_dev, folds_optimal_epochs = get_average_results_with_stdev(folds_dictionary, "test acc")

    # plot optimal epochs
    folds_optimal_epochs = dict(sorted(folds_optimal_epochs.items()))
    plt.figure()
    plt.bar(range(len(folds_optimal_epochs)), list(folds_optimal_epochs.values()), align='center')
    plt.xticks(range(len(folds_optimal_epochs)), list(folds_optimal_epochs.keys()))
    plt.title("Optimal number of epochs (best validation F1 score)")
    plt.ylabel("Epoch")
    plt.xlabel("Fold")
    plt.savefig(folder_results + "optimal_epochs.pdf")
    plt.close()

    print("Results ")
    print(f"F1: \t {avg_f1} \t (+/- {f1_std_dev})")
    print(f"Prec: \t {avg_prec} \t (+/- {prec_std_dev})")
    print(f"Rec: \t {avg_rec} \t (+/- {rec_std_dev})")
    print(f"Acc: \t {avg_acc} \t (+/- {acc_std_dev})")


if __name__ == '__main__':
    folder_results = "../../sarcasm_v2_experiments/results/GPT_2_lr0.02/" #sys.argv[1]
    evaluate_results(folder_results)
