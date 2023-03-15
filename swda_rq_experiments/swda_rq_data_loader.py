"""
@author White Wolf
"""

from swda_rq_experiments.swda_rq import SwDA_RQ

# FORMAT
# {0,1},utterance_i {%,&} utterance_i+1 {%,&}-1 utterance_i-1 {%,&}-2 utterance_i-2
#
# FORMAT-DETAILS
# 0: non-rhetorical
# 1: rhetorical
#
# &, &-1, &-2 : The given utterance is spoken by the same person who spoke the main utterance.
# %, %-1, %-2 : otherwise.
#
# t_con: connective
# t_laugh: laugh
# t_empty: no utterance (e.g. "% t_empty" means no utterance follows the main utterance.)

# 1,well, when is the party here? t_laugh . % Yeah. &-1 Jeez, %-2 How does that sound? t_laugh .


def load_preprocessed_swda_data(dataset):
    swda_rq_data = []

    with open(f"{__file__[:-len('swda_rq_experiments/swda_rq_data_loader.py')]}rhetorical_questions_data/" + dataset + ".txt", encoding="utf8", mode="r") as fr:
        lines = fr.readlines()
    for line in lines:
        label = line[0]
        text = line[2:]

        # print("Line: ", text)
        position = 0
        if " % " in text:
            position = text.find(" % ") + 3
            current_utterance = text.split(" % ")[0]
        if " & " in text:
            current_utterance = text.split(" & ")[0]
            position = text.find(" % ") + 3

        # print("Current utterance: ", current_utterance)

        text = text[position:]
        # print(text)

        if " %-1 " in text:
            position = text.find(" %-1 ") + 4
            next_utterance = text.split(" %-1 ")[0]
        if " &-1 " in text:
            next_utterance = text.split(" &-1 ")[0]
            position = text.find(" &-1 ") + 4

        # print("Next utterance: ", next_utterance)

        text = text[position:]
        # print(text)

        if " %-2 " in text:
            position = text.find(" %-2 ") + 4
            previous_utterance = text.split(" %-2 ")[0]
        if " &-2 " in text:
            previous_utterance = text.split(" &-2 ")[0]
            position = text.find(" &-2 ") + 4

        previous_utterance_2 = text[position:].strip()

        swda_rq_item = SwDA_RQ(
            gold_label = int(label),
            previous_utterance_2=previous_utterance_2,
            previous_utterance=previous_utterance,
            current_utterance=current_utterance,
            next_utterance=next_utterance
        )

        swda_rq_data.append(swda_rq_item)

        # input_texts.append(current_utterance + " " + next_utterance)
        # input_texts.append(previous_utterance_2 + " " + previous_utterance + " " + current_utterance + " " + next_utterance)
        # print(line)
        # print(previous_utterance_2 + " " + previous_utterance + " " + current_utterance + " " + next_utterance)
        # exit(0)

        #labels.append(int(label))
        # next_utterance =""
        # previous_utterance =""
        # previsou_utterance_2 =""

    return swda_rq_data