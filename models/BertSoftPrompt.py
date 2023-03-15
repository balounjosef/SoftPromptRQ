import torch
from transformers import BertModel, BertForNextSentencePrediction
from models.AbstractSoftPrompt import AbstractSoftPromptModel




class BertForNextSentencePredictionSoftPromptModel(AbstractSoftPromptModel):

    def __init__(self, modelid, device=None, prompts=None, prompt_embed_text_init=None, bcehead=False, variational_prompts=False):
        model = BertForNextSentencePrediction.from_pretrained(modelid)
        super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device, prompts=prompts,
                         classes=1 if bcehead else 2, prompt_embed_text_init=prompt_embed_text_init, tokenizerid=modelid,
                         variational_prompts=variational_prompts, own_cls_head=False)
        self.bcehead = bcehead

    def forward(self, input_ids, attention_mask):
        batch = len(input_ids)

        #   prepare the input embeddings and attention mask
        embeds = self.model.get_input_embeddings()(input_ids.to(self.device))  # pretrained input embeddings of frozen model
        prompts = self.get_prompts(batch)  # our learnable prompts
        input_embeds = torch.cat([embeds, prompts], axis=1)  # ordering of the input tokens
        # CLS token is the first one

        # preparation of attention mask for input_embeds - prompts are not masked => 1, input mask as is
        prompts_mask = torch.ones((batch, self.numprompts), dtype=attention_mask.dtype)
        input_embeds_att_mask = torch.cat([attention_mask, prompts_mask], axis=1).to(self.device)

        x = self.model(inputs_embeds=input_embeds,
                       attention_mask=input_embeds_att_mask)  # forward with prepared embedding and mask

        # default head uses softmax
        if self.bcehead:
            res = x.logits[:, 0] - x.logits[:, 1]  # softmax to sigmoid: pos - negative logits
            return res.reshape((-1, 1))

        return x.logits


class BertSoftPromptTripletCLSModel(AbstractSoftPromptModel):

    def __init__(self, modelid, device=None, prompts=None, prompt_embed_text_init=None, variational_prompts=False, out_dim=None):
        model = BertModel.from_pretrained(modelid)
        super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device, prompts=prompts,
                         classes=out_dim, prompt_embed_text_init=prompt_embed_text_init, tokenizerid=modelid,
                         variational_prompts=variational_prompts, own_cls_head=False)

        if out_dim is None:
            self.final_head = lambda x: x
        else:
            self.final_head = torch.nn.Sequential(    # learnable classifier head
                torch.nn.Linear(model.config.hidden_size, out_dim)
            )
            self.final_head.to(self.device)

    def forward(self, input_ids, attention_mask):
        batch = len(input_ids)

        #   prepare the input embeddings and attention mask
        embeds = self.model.get_input_embeddings()(input_ids.to(self.device))  # pretrained input embeddings of frozen model
        prompts = self.get_prompts(batch)  # our learnable prompts
        input_embeds = torch.cat([embeds, prompts], axis=1)  # ordering of the input tokens
        # CLS token is the first one

        # preparation of attention mask for input_embeds - prompts are not masked => 1, input mask as is
        prompts_mask = torch.ones((batch, self.numprompts), dtype=attention_mask.dtype)
        input_embeds_att_mask = torch.cat([attention_mask, prompts_mask], axis=1).to(self.device)

        x = self.model(inputs_embeds=input_embeds, attention_mask=input_embeds_att_mask)  # forward with prepared embedding and mask

        # cls token
        cls = x.last_hidden_state[:, 0]
        return self.final_head(cls)


if __name__ == '__main__':
    from models.utils.model_utils import test_overfit
    from transformers import BertTokenizerFast

    modelid = "bert-base-cased"
    tokenizer = BertTokenizerFast.from_pretrained(modelid)
    model = BertSoftPromptTripletCLSModel(modelid, prompts=2, out_dim=2)
    test_overfit(model, tokenizer)
