"""
@author White Wolf
"""

class RquetItem:

    def __init__(self, ID, ctx_before2, ctx_before2_speaker, ctx_before1, ctx_before1_speaker, question, question_speaker, ctx_after1, ctx_after1_speaker, ctx_after2, ctx_after2_speaker, gold_label):
        self.ID = ID
        self.ctx_before2 = ctx_before2
        self.ctx_before2_speaker = ctx_before2_speaker
        self.ctx_before1 = ctx_before1
        self.ctx_before1_speaker = ctx_before1_speaker
        self.question = question
        self.question_speaker = question_speaker
        self.ctx_after1 = ctx_after1
        self.ctx_after1_speaker = ctx_after1_speaker
        self.ctx_after2 = ctx_after2
        self.ctx_after2_speaker = ctx_after2_speaker
        self.gold_label = gold_label

        # if a question speaker differs from ctx_after1
        if self.question_speaker == self.ctx_after1_speaker:
            self.speaker_change_after_question = False
        else:
            self.speaker_change_after_question = True


