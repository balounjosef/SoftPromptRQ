import torch
from transformers import GPT2Model
from models.AbstractSoftPrompt import AbstractSoftPromptModel

# # USE AbstractSoftPromptModel directly
# model = GPT2Model.from_pretrained(modelid)
# AbstractSoftPromptModel(model=model, modelid=modelid, embed_dim=model.config.n_embd, prompts=prompts, classes=classes, prompt_embed_text_init=prompt_embed_text_init, tokenizerid=modelid)


class GPT2SoftPromptTripletCLSModel(AbstractSoftPromptModel):

    def __init__(self, modelid, device=None, prompts=None, prompt_embed_text_init=None, variational_prompts=False, out_dim=None):
        model = GPT2Model.from_pretrained(modelid)
        super().__init__(model=model, modelid=modelid, embed_dim=model.config.n_embd, device=device, prompts=prompts,
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
        input_embeds = torch.cat([prompts, embeds], axis=1)  # ordering of the input tokens
        # CLS token is the last one

        # preparation of attention mask for input_embeds - prompts are not masked => 1, input mask as is
        prompts_mask = torch.ones((batch, self.numprompts), dtype=attention_mask.dtype)
        input_embeds_att_mask = torch.cat([prompts_mask, attention_mask], axis=1).to(self.device)

        x = self.model(inputs_embeds=input_embeds, attention_mask=input_embeds_att_mask)  # forward with prepared embedding and mask

        # cls token
        cls = x.last_hidden_state[:, -1]
        return self.final_head(cls)


if __name__ == '__main__':
    from transformers import GPT2TokenizerFast
    from models.utils.model_utils import test_overfit

    modelid = "gpt2"
    model = GPT2SoftPromptTripletCLSModel(modelid, prompt_embed_text_init="hi there")
    tokenizer = GPT2TokenizerFast.from_pretrained(modelid, truncation_side='left', padding_side="left")
    test_overfit(model, tokenizer)


