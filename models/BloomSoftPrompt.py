# from transformers import BloomModel
# from models.AbstractSoftPrompt import AbstractSoftPromptModel
#
# # USE AbstractSoftPromptModel directly
# class BloomSoftPromptModel(AbstractSoftPromptModel):
#
#     def __init__(self, modelid, device=None, prompts=20, classes=1):
#         model = BloomModel.from_pretrained(modelid)
#         super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device, prompts=prompts, classes=classes)
#
#
# if __name__ == '__main__':
#     from transformers import BloomTokenizerFast
#     from models.utils.model_utils import test_overfit
#
#     modelid = "bigscience/bloom-560m"
#     model = BloomSoftPromptModel(modelid)
#     tokenizer = BloomTokenizerFast.from_pretrained(modelid, truncation_side='left', padding_side='left')
#     test_overfit(model, tokenizer, lr=0.0002)
