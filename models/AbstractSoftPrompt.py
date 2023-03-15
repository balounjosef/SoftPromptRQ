from torch.nn import Parameter
from models.utils.model_utils import freeze_model, decide_device, decide_tokenizer
import torch
from torch import nn


class AbstractSoftPromptModel(nn.Module):
    """
    Intended use is to wrap any ENC or DEC based LM, freeze it, add own learnable prompts at the beginning and cls at the end and finally classification head from cls.
    Example
    model = BertModel.from_pretrained(modelid)
    AbstractSoftPromptModel(model=model, modelid=modelid, embed_dim=model.config.hidden_size, prompts=10, classes=1)
    """
    def __init__(self, model, modelid, embed_dim=None, prompt_embed_text_init=None, tokenizerid=None,
                 prompts=None, variational_prompts=False,
                 classes=1, own_cls_head=True,
                 device=None
                 ):
        """
        Parameters
        ----------
        model - Pretrained lang. model to use
        modelid - modelid used (for logging purposes)
        embed_dim - embedding dimension (for random initialization)
        prompt_embed_text_init - str prompt to initialize embeddings (for text initialization)
        tokenizerid - tokenizer id to use (for text initialization)
        prompts - number of prompts (for random initialization)
        variational_prompts - bool if variational prompts should be used
        classes - number of output classes
        own_cls_head - bool if own cls token and class. head should be created
        device - cpu/cuda torch device
        """
        super().__init__()
        self.model = model
        freeze_model(self.model)    # freeze the whole model so the weights and word embeddings wil not be updated

        self.device = decide_device(device)
        self.numclasses = classes
        self.modelid = modelid
        self.variational_prompts = variational_prompts
        self.own_cls_head = own_cls_head

        if prompt_embed_text_init is not None and tokenizerid is not None:      # initialize prompts with embeddings
            tokenizer, _, _ = decide_tokenizer(tokenizerid)
            prompt_init_embed_ids = tokenizer(prompt_embed_text_init, return_tensors="pt").input_ids
            print(f"{prompt_init_embed_ids.shape[1]} prompts (embeddings) for text '{prompt_embed_text_init}'")

            ie = self.model.get_input_embeddings()(prompt_init_embed_ids)[0].clone()
            self.prompts = Parameter(ie)
            assert len(self.prompts.shape) == 2
            
            self.numprompts = self.prompts.shape[0]
            self.embed_dim = self.prompts.shape[1]
            self.prompt_embed_text_init = prompt_embed_text_init
        elif embed_dim is not None and prompts is not None:       # initialize prompts randomly
            self.numprompts = prompts
            self.embed_dim = embed_dim
            self.prompt_embed_text_init = None
            self.prompts = Parameter(torch.empty((self.numprompts, self.embed_dim)))  # learnable prompt tokens as matrix of trainable weights
            nn.init.xavier_uniform_(self.prompts)
        else:
            self.numprompts = None
            self.embed_dim = embed_dim
            self.prompt_embed_text_init = None
            self.prompts = None

        if self.variational_prompts:
            self.prompts_std = Parameter(torch.ones((self.numprompts, self.embed_dim)))  # for distribution
        else:
            self.prompts_std = None

        if self.own_cls_head:
            self.clsprompt = Parameter(torch.empty((1, self.embed_dim)))              # learnable cls token
            nn.init.xavier_uniform_(self.clsprompt)

            self.classifier = nn.Sequential(    # learnable classifier head
                nn.Linear(self.embed_dim, self.numclasses)
            )
        else:
            self.clsprompt = None
            self.classifier = None

        self.to(self.device)

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

        #   preparation of input_embeds (batch_size, sequence_length, hidden_size)
        #   position embeddings are added in forward method: hidden_states = inputs_embeds + position_embeds (GPT)
        #   so only prepare the input embeddings and attention mask
        embeds = self.model.get_input_embeddings()(input_ids.to(self.device))  # pretrained input embeddings of frozen model
        prompts = self.get_prompts(batch)    # our learnable prompts
        clsprompt = self.clsprompt.expand(batch, -1, -1)    # our learnable cls token
        input_embeds = torch.cat([prompts, embeds, clsprompt], axis=1)  # ordering of the input tokens

        # preparation of attention mask for input_embeds - prompts and CLS are not masked => 1, input mask as is
        prompts_mask = torch.ones((batch, self.numprompts), dtype=attention_mask.dtype)
        cls_mask = torch.ones((batch, 1), dtype=attention_mask.dtype)
        input_embeds_att_mask = torch.cat([prompts_mask, attention_mask, cls_mask], axis=1).to(self.device)

        x = self.model(inputs_embeds=input_embeds, attention_mask=input_embeds_att_mask)    # forward with prepared embedding and mask
        cls = x.last_hidden_state[:, -1, :]     # taking only CLS token (last token in last hidden state)
        return self.classifier(cls)    # classification based on CLS

    def get_prompts(self, batch):
        if self.variational_prompts and self.training:
            mean = self.prompts.expand(batch, -1, -1)
            std = self.prompts_std.expand(batch, -1, -1)
            rnd = torch.randn(mean.shape).to(self.device)
            return (rnd * std) + mean
        else:
            return self.prompts.expand(batch, -1, -1)

    def save_learnable_params(self, path):
        sd = self.state_dict()
        rmkeys = [name for name, param in self.named_parameters() if not param.requires_grad]
        for k in rmkeys:
            sd.pop(k)
        torch.save(sd, path)

    def load_learnable_params(self, path):
        mk = self.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        # for k in mk.missing_keys:
        #     assert k.startswith("model.")

    def get_num_of_learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    from transformers import BloomModel
    model = BloomModel.from_pretrained("bigscience/bloom-560m")
    model = AbstractSoftPromptModel(model, "bigscience/bloom-560m", model.config.hidden_size, prompts=2)

    model.get_num_of_learnable_params()

    model.save_learnable_params("tmp.sd")

    model.load_learnable_params("tmp.sd")

