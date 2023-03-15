
class Mrda():
    def __init__(self, utterance, basic_label, general_label, full_label):
        self.utterance = utterance
        self.basic_label = basic_label
        self.general_label = general_label
        self.full_label = full_label
        self.basic_label_list = ["S", "B", "D", "F", "Q"]