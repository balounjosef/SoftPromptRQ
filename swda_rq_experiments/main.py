

import swda_rq_data_loader


swda_rq_data_train = swda_rq_data_loader.load_preprocessed_swda_data("train")
print(len(swda_rq_data_train))

swda_rq_data_test = swda_rq_data_loader.load_preprocessed_swda_data("test")
print(len(swda_rq_data_test))
print(swda_rq_data_test[6].next_utterance)

