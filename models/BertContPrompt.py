import math

import torch
from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertSelfAttention

from models.AbstractSoftPrompt import AbstractSoftPromptModel


class OwnBertSelfAttention(BertSelfAttention):

    def __init__(self, parentinstance, config, numprompts=1):
        # super().__init__(config)
        self.__dict__ = parentinstance.__dict__.copy()

        self.prompts = torch.nn.Parameter(torch.empty((numprompts, config.hidden_size)))
        torch.nn.init.xavier_uniform_(self.prompts)
        self.prompts_mask = torch.ones((1, 1, numprompts), dtype=torch.float32)

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        batch = hidden_states.shape[0]
        prompts = self.prompts.expand(batch, -1, -1)
        input_embeds = torch.cat([hidden_states, prompts], axis=1)  # ordering of the input tokens

        # preparation of attention mask for input_embeds - prompts and CLS are not masked => 1, input mask as is
        prompts_mask = self.prompts_mask.expand(batch, -1, -1, -1).to(attention_mask.device)
        input_embeds_att_mask = torch.cat([attention_mask, prompts_mask], axis=-1)

        res = super().forward(input_embeds, input_embeds_att_mask, *args, **kwargs)
        # res[0] = res[0][:,:-1]
        return (res[0][:, :-self.prompts.shape[0]],)


class BertContPrompt(AbstractSoftPromptModel):

    def __init__(self, modelid, prompts=1, device=None, out_dim=None):
        model = BertModel.from_pretrained(modelid)
        for l in range(len(model.base_model.encoder.layer)):
            model.base_model.encoder.layer[l].attention.self = OwnBertSelfAttention(
                model.base_model.encoder.layer[l].attention.self, model.base_model.config, prompts)

        super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device,
                         prompts=None,
                         classes=None, prompt_embed_text_init=None,
                         tokenizerid=modelid,
                         variational_prompts=False, own_cls_head=False)

        for l in self.model.base_model.encoder.layer:
            l.attention.self.prompts.requires_grad = True

        self.prompts = [self.model.base_model.encoder.layer[l].attention.self.prompts for l in
                        range(len(model.encoder.layer))]
        self.numprompts = len(self.prompts) * prompts

        if out_dim is None:
            self.final_head = lambda x: x
        else:
            self.final_head = torch.nn.Sequential(    # learnable classifier head
                torch.nn.Linear(model.config.hidden_size, out_dim)
            )
            self.final_head.to(self.device)

    @property
    def prompts(self):
        return torch.cat(self.__prompts, dim=0)

    @prompts.setter
    def prompts(self, p):
        self.__prompts = p

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids.to(self.device),
                       attention_mask=attention_mask.to(self.device))  # forward with prepared embedding and mask

        # cls token
        cls = x.last_hidden_state[:, 0]
        return self.final_head(cls)


class OwnBertSelfAttentionKV(BertSelfAttention):

    def __init__(self, parentinstance, config, numprompts=1):
        # super().__init__(config)
        self.__dict__ = parentinstance.__dict__.copy()
        shp = (config.num_attention_heads, numprompts, config.hidden_size // config.num_attention_heads)
        self.promptsK = torch.nn.Parameter(torch.empty(shp))
        torch.nn.init.xavier_uniform_(self.promptsK)
        self.promptsV = torch.nn.Parameter(torch.empty(shp))
        torch.nn.init.xavier_uniform_(self.promptsV)

        # mask is 0 or -infinity
        self.prompts_mask = torch.zeros((1, 1, numprompts), dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # (batch, heads, seqlen, dimhead)

        ################# EDITED K, V, AttentionMask
        batch = attention_mask.shape[0]
        key_layer = torch.cat([key_layer, self.promptsK.expand(batch, -1, -1, -1)], dim=-2)
        value_layer = torch.cat([value_layer, self.promptsV.expand(batch, -1, -1, -1)], dim=-2)

        attention_mask = torch.cat([attention_mask, self.prompts_mask.expand(batch, -1, -1, -1).to(attention_mask.device)], dim=-1)
        #################

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        ######## COMMENTED

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertContPromptKV(AbstractSoftPromptModel):

    def __init__(self, modelid, prompts=1, device=None, out_dim=None):
        model = BertModel.from_pretrained(modelid)
        for l in range(len(model.base_model.encoder.layer)):
            model.base_model.encoder.layer[l].attention.self = OwnBertSelfAttentionKV(
                model.base_model.encoder.layer[l].attention.self, model.base_model.config, prompts)

        super().__init__(model=model, modelid=modelid, embed_dim=model.config.hidden_size, device=device,
                         prompts=None,
                         classes=None, prompt_embed_text_init=None,
                         tokenizerid=modelid,
                         variational_prompts=False, own_cls_head=False)

        for l in self.model.base_model.encoder.layer:
            l.attention.self.promptsK.requires_grad = True
            l.attention.self.promptsV.requires_grad = True

        self.prompts = [self.model.base_model.encoder.layer[l].attention.self.promptsK for l in range(len(model.encoder.layer))] +\
            [self.model.base_model.encoder.layer[l].attention.self.promptsV for l in range(len(model.encoder.layer))]
        self.numprompts = len(self.prompts) * prompts

        if out_dim is None:
            self.final_head = lambda x: x
        else:
            self.final_head = torch.nn.Sequential(    # learnable classifier head
                torch.nn.Linear(model.config.hidden_size, out_dim)
            )
            self.final_head.to(self.device)

    @property
    def prompts(self):
        return torch.cat(self.__prompts, dim=0)

    @prompts.setter
    def prompts(self, p):
        self.__prompts = p

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids=input_ids.to(self.device),
                       attention_mask=attention_mask.to(self.device))  # forward with prepared embedding and mask

        # cls token
        cls = x.last_hidden_state[:, 0]
        return self.final_head(cls)


if __name__ == '__main__':
    from models.utils.model_utils import test_overfit
    from transformers import BertTokenizerFast

    modelid = "bert-base-cased"
    tokenizer = BertTokenizerFast.from_pretrained(modelid)
    model = BertContPromptKV(modelid, prompts=1, out_dim=1)
    test_overfit(model, tokenizer, lr=0.1)
