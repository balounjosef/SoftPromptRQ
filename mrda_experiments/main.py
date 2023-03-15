"""
@author White Wolf
"""
import mrda_data_loader

if __name__ == '__main__':
    print("ISQ/NISQ question experiments")
    dataset = "train"
    train_rquet_data = mrda_data_loader.load_mrda_data(dataset)
    print("Successfully loaded", len(train_rquet_data), dataset," rquet data")

    print(train_rquet_data[0].utterance)
    print(train_rquet_data[0].basic_label)
    print(train_rquet_data[0].general_label)
    print(train_rquet_data[0].full_label)
