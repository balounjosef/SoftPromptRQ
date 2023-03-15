"""
@author White Wolf
"""
import rquet_data_loader

if __name__ == '__main__':
    print("ISQ/NISQ question experiments")
    dataset = "train"
    test_rquet_data = rquet_data_loader.load_rquet_data(dataset)
    print("Successfully loaded", len(test_rquet_data), dataset," rquet data")
    print(test_rquet_data[13].ID)
    print(test_rquet_data[13].question)
    print(test_rquet_data[13].question_speaker)
    print(test_rquet_data[13].ctx_after1_speaker)
    print(test_rquet_data[13].speaker_change_after_question)

