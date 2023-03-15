"""
@author White Wolf
"""


import numpy as np
from matplotlib import pyplot as plt

import swda_rq_data_loader


def get_label_distribution(swda_rq_data):
    labels = []
    for data in swda_rq_data:
        labels.append(data.gold_label)
    labels_array = np.array(labels)
    print("Rhet: ", np.count_nonzero(labels_array), 100 * (np.count_nonzero(labels_array) / len(labels_array)), "%")
    print("NonRhet: ", len(labels_array) - np.count_nonzero(labels_array))

def get_average_text_length(swda_rq_data):
    text_lengts = []
    for data in swda_rq_data:
        text = data.previous_utterance_2 + data.previous_utterance + data.current_utterance
        text_lengts.append(len(text.split()))

    print("Average text length: ", np.average(np.array(text_lengts)))
    print("Max text length: ", np.max(np.array(text_lengts)))


    plt.figure()
    plt.boxplot(np.array(text_lengts))
    plt.show()


if __name__ == '__main__':
    swda_rq_data_train = swda_rq_data_loader.load_preprocessed_swda_data("train1")
    print(len(swda_rq_data_train))
    swda_rq_data_validate = swda_rq_data_loader.load_preprocessed_swda_data("validate1")
    print(len(swda_rq_data_validate))
    swda_rq_data_test = swda_rq_data_loader.load_preprocessed_swda_data("test")
    print(len(swda_rq_data_test))

    # average text alength for all cca 18 words
    get_average_text_length(swda_rq_data_train)
    get_average_text_length(swda_rq_data_validate)
    get_average_text_length(swda_rq_data_test)

    # 5 percent of the question are rhetorical
    get_label_distribution(swda_rq_data_train)
    get_label_distribution(swda_rq_data_validate)
    get_label_distribution(swda_rq_data_test)