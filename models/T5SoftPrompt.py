from transformers import T5Model, T5TokenizerFast
from models.AbstractSoftPrompt import AbstractSoftPromptModel
import torch


class T5SoftPromptModel(AbstractSoftPromptModel):

    def __init__(self, modelid, device=None, prompts=20, out_dim=1):
        model = T5Model.from_pretrained(modelid)

        # T5 uses the pad_token_id as the starting token for decoder_input_ids
        pad_token_id = T5TokenizerFast.from_pretrained(modelid).pad_token_id

        # super().__init__(model=model, modelid=modelid, embed_dim=model.config.d_model, device=device, prompts=prompts, classes=classes)
        super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device,
                         prompts=prompts,
                         classes=out_dim, tokenizerid=modelid, own_cls_head=False)

        if out_dim is None:
            self.final_head = lambda x: x
        else:
            self.final_head = torch.nn.Sequential(  # learnable classifier head
                torch.nn.Linear(model.config.hidden_size, out_dim)
            )
            self.final_head.to(self.device)

        self.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask):
        """
        input_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of input sequence tokens in the vocabulary.
        attention_mask (torch.FloatTensor of shape (batch_size, sequence_length)) — Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked.

        maximal allowed input sequence length is lowered due to prompts and CLS token
        it is given by: self.model.config.n_positions - self.numprompts - 1
        """
        batch = len(input_ids)

        # preparation of input_embeds (batch_size, sequence_length, hidden_size)
        embeds = self.model.get_input_embeddings()(input_ids.to(self.device))  # pretrained input embeddings of frozen GPT
        prompts = self.prompts.expand(batch, -1, -1)    # our learnable prompts
        input_embeds = torch.cat([prompts, embeds], axis=1)  # ordering of the input tokens

        # preparation of attention mask for input_embeds - prompts are not masked =>
        prompts_mask = torch.ones((batch, self.numprompts), dtype=attention_mask.dtype)
        input_embeds_att_mask = torch.cat([prompts_mask, attention_mask], axis=1).to(self.device)

        # decoder_input_ids (torch.LongTensor of shape (batch_size, target_sequence_length), optional) — Indices of decoder input sequence tokens in the vocabulary.
        decoder_input_ids = torch.full((batch, 1), self.pad_token_id, dtype=torch.long).to(self.device)

        x = self.model(inputs_embeds=input_embeds, attention_mask=input_embeds_att_mask, decoder_input_ids=decoder_input_ids)    # forward with prepared embedding and mask
        # x = self.model(inputs_embeds=input_embeds, attention_mask=input_embeds_att_mask, decoder_input_ids=torch.ones((2,1), dtype=torch.long))

        cls = x.last_hidden_state[:, 0, :]     # taking cls token
        return self.final_head(cls)    # classification based on CLS


if __name__ == '__main__':
    from transformers import T5TokenizerFast
    from models.utils.model_utils import test_overfit

    # modelid = "google/t5-v1_1-small"
    modelid = "google/flan-t5-small"
    model = T5SoftPromptModel(modelid, prompts=5)
    tokenizer = T5TokenizerFast.from_pretrained(modelid)
    test_overfit(model, tokenizer, lr=0.2)
