"""
@author White Wolf
"""
import rquet_data_loader
import numpy as np
import matplotlib.pyplot as plt

def get_average_text_length(rquet_data):
    text_lengts = []
    for data in rquet_data:
        text = str(data.ctx_before2) + " " + str(data.ctx_before1) + " " + str(data.question)
        text_lengts.append(len(text.split()))
    print("Average text length: ", np.average(np.array(text_lengts)))
    print("Max text length: ", np.max(np.array(text_lengts)))


    plt.figure()
    plt.boxplot(np.array(text_lengts))
    plt.show()


def analyze_speaker_changes(rquet_data):
    csv_file = open("speaker_changes_analysis.csv", mode="w", encoding="utf8")
    speakers_changes = []
    uncorrect = 0
    for rquet_item in rquet_data:
        all_speakers = rquet_item.ctx_before2_speaker, rquet_item.ctx_before1_speaker, rquet_item.question_speaker, rquet_item.ctx_after1_speaker, rquet_item.ctx_after2_speaker
        # print(all_speakers)
        all_speakers_string = str(rquet_item.ID) + str(rquet_item.ctx_before2_speaker)+ str(rquet_item.ctx_before1_speaker) + str(rquet_item.question_speaker) + str(rquet_item.ctx_after1_speaker) + str(rquet_item.ctx_after2_speaker)

        number_of_lower_cased_letters = sum(1 for c in all_speakers_string if c.islower())
        # due to the tsv format, some speaker might miss in data --> instead of SPEAKER is lowercased parts of dialogue
        if number_of_lower_cased_letters > 100:
            # problematic row
            uncorrect += 1
            #print(all_speakers)
        else:
            # highly probably a correct records
            print(all_speakers)
            # we create a set from all_speakers touple to find out number of diff speakers
            speaker_list = []
            for name in all_speakers:
                if name not in speaker_list:
                    speaker_list.append(name)
            number_of_diff_speakers = len(speaker_list)
            encoded_all_speakers = []
            for name in all_speakers:
                index = speaker_list.index(name)
                encoded_all_speakers.append(index)
            print(encoded_all_speakers)
            speakers_changes.append(encoded_all_speakers)
            print()

            csv_file.write(str(encoded_all_speakers).replace("[", "").replace("]", "").replace(" ", "") + "," + str(rquet_item.gold_label)+"\n")

    print("Problematic Speaker change:", uncorrect, "/" ,len(rquet_data))
    print("Problematic Speaker change:", round(((uncorrect / len(rquet_data)) * 100),2), "%")

    csv_file.close()

if __name__ == '__main__':
    print("ISQ/NISQ question experiments")
    dataset = "train"
    train_rquet_data = rquet_data_loader.load_rquet_data(dataset)
    print("Successfully loaded", len(train_rquet_data), dataset, " rquet data")
    #analyze_speaker_changes(train_rquet_data)
    get_average_text_length(train_rquet_data)

    dataset = "test"
    test_rquet_data = rquet_data_loader.load_rquet_data(dataset)
    print("Successfully loaded", len(test_rquet_data), dataset, " rquet data")
    #analyze_speaker_changes(test_rquet_data)
    get_average_text_length(test_rquet_data)

