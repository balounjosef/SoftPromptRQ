"""
@author White Wolf
"""
import pandas as pd

import paths
from mrda_experiments.mrda import Mrda


def load_mrda_data(dataset):
    with open(paths.root + "mrda_data/" + dataset + "_set.txt", "r", encoding="utf-8") as file:
        mrda_data = file.readlines()


    mrda_items = []
    for line in mrda_data:
        line_elements = line.split("|")
        utterance = line_elements[1].strip()
        basic_label = line_elements[2].strip()
        general_label = line_elements[3].strip()
        full_label = line_elements[4].strip()
        mrda_item = Mrda(utterance, basic_label, general_label, full_label)
        mrda_items.append(mrda_item)
    return mrda_items