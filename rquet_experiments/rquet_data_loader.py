"""
@author White Wolf
"""
import pandas as pd
from rquet_experiments.rquet import RquetItem


def load_rquet_data(dataset):
    """
    loads rquet data, just a question and ctx_after1 for now
    :param dataset: train or test
    :return: list of input_text and labels
    """
    dataframe = pd.read_csv(f"{__file__[:-len('rquet_experiments/rquet_data_loader.py')]}rquet_dataset/rquet_" + dataset + "set.csv", sep='\t')
    rquet_data = []
    # print(dataframe.head)
    for index in dataframe.index:
        rquet_item = RquetItem(
                        dataframe['ID'][index],
                        dataframe['ctx_before2'][index],
                        dataframe['ctx_before2_speaker'][index],
                        dataframe['ctx_before1'][index],
                        dataframe['ctx_before1_speaker'][index],
                        dataframe['question'][index],
                        dataframe['question_speaker'][index],
                        dataframe['ctx_after1'][index],
                        dataframe['ctx_after1_speaker'][index],
                        dataframe['ctx_after2'][index],
                        dataframe['ctx_after2_speaker'][index],
                        dataframe['gold_label'][index]
        )
        rquet_data.append(rquet_item)
    return rquet_data