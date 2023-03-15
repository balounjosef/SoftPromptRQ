"""
@author White Wolf
"""

from sklearn.model_selection import KFold
import pandas as pd
from sarcasm_v2_experiments.sarcasm_v2 import Sarcasm_v2


def load_sarcasm_v2_data(dataset, folds):
    if dataset != "GEN" and dataset != "HYP" and dataset != "RQ":
        print("Unknown dataset type: list of supported [GEN, HYP, RQ]")
        exit(1)

    dataframe = pd.read_csv(f"{__file__[:-len('sarcasm_v2_experiments/sarcasm_v2_data_loader.py')]}sarcasm_v2/" + dataset + "-sarc-notsarc.csv", sep=',')
    sarcasm_v2_data = {}
    sarc_data, not_sarc_data = [], []
    for index in dataframe.index:
        sarcasm_v2_item = Sarcasm_v2(
                        dataframe['class'][index],
                        dataframe['id'][index],
                        dataframe['text'][index]
        )
        sarcasm_v2_data[dataframe['id'][index]] = sarcasm_v2_item
        if sarcasm_v2_item.gold_label == "sarc":
            sarc_data.append(sarcasm_v2_item)
        else:
            not_sarc_data.append(sarcasm_v2_item)


    train_folds, test_folds = prepare_crossvalidation(sarc_data, not_sarc_data, folds)
    return sarcasm_v2_data, train_folds, test_folds


def prepare_crossvalidation(sarc_data, not_sarc_data, k=10):
    """
    Function creates folds according to the param k, returns train a test folds. We need balanced folds,that's why we
    firstly split dataset int sarc and not sarc
    :param sarc_data list of sarc data
    :param not_sarc_data of not-sarc data
    :param k: how many folds
    :return: train and test folds
    """

    sarcasm_v2_all_data = []

    train_sarc_folds, test_sarc_folds = [], []
    train_not_sarc_folds, test_not_sarc_folds = [], []

    kfold = KFold(n_splits=k, shuffle=False)
    # enumerate splits
    fold_counter = 1

    # not sarc data
    for train_not_sarc, test_not_sarc in kfold.split(not_sarc_data):
        train_indices = []
        for index in train_not_sarc:
            sarcasm_v2_item = not_sarc_data[index]
            train_indices.append(sarcasm_v2_item.id)
        test_indices = []
        for index in test_not_sarc:
            sarcasm_v2_item = not_sarc_data[index]
            test_indices.append(sarcasm_v2_item.id)

        train_not_sarc_folds.append(train_indices)
        test_not_sarc_folds.append(test_indices)

        fold_counter += 1

    kfold = KFold(n_splits=k, shuffle=False)
    # sarc data
    for train_sarc, test_sarc in kfold.split(sarc_data):
        train_indices = []
        for index in train_sarc:
            sarcasm_v2_item = sarc_data[index]
            train_indices.append(sarcasm_v2_item.id)

        test_indices = []
        for index in test_sarc:
            sarcasm_v2_item = sarc_data[index]
            test_indices.append(sarcasm_v2_item.id)

        train_sarc_folds.append(train_indices)
        test_sarc_folds.append(test_indices)

        fold_counter += 1

    train_folds, test_folds = train_not_sarc_folds, test_not_sarc_folds
    i = 0
    for tns in train_sarc_folds:
        for sample in tns:
            # sample is item.id
            train_folds[i].append(sample)
        i += 1

    i = 0
    for tns in test_sarc_folds:
        for sample in tns:
            # sample is item.id
            test_folds[i].append(sample)
        i += 1

    return train_folds, test_folds

