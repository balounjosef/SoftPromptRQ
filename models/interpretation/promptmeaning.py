import math
import torch
import numpy as np


class EditedBertSelfAttention(torch.nn.Module):
    def __init__(self, num_attention_heads, attention_head_size, query, key):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.query = query
        self.key = key

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, prompts, embeds):
        key_layer = self.transpose_for_scores(self.key(embeds))
        query_layer = self.transpose_for_scores(self.query(prompts))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        return attention_probs


def bert_find_similiar_embeddings(model, tokenizer, layernorm=True):
    print(f"PROMPT (Q) - EMBED (K) scaled dot-product attention (layernorm={layernorm})")
    if model.clsprompt is not None:
        prompts = torch.cat((model.prompts, model.clsprompt), dim=0)
    else:
        prompts = model.prompts
    embs = model.model.base_model.get_input_embeddings().weight
    prompts = prompts.reshape((1,) + prompts.shape)
    embs = embs.reshape((1,) + embs.shape)

    if layernorm:
        tmp = model.model.base_model.embeddings.LayerNorm(torch.cat((prompts, embs), dim=1))
        prompts = tmp[:, :prompts.shape[1]]
        embs = tmp[:, prompts.shape[1]:]

    # destroy model a bit :)
    enclayeratt = model.model.base_model.encoder.layer[0].attention.self     # only first layer attention interests us
    edtatt = EditedBertSelfAttention(enclayeratt.num_attention_heads, enclayeratt.attention_head_size, enclayeratt.query, enclayeratt.key)
    edtatt.train(False)

    atts = edtatt(prompts, embs) # attentions of first layer per each head
    atts = torch.max(atts, dim=1).values    # max value from all heads
    atts = atts[0]  # ignore batch
    # atts = atts[:noprompts, noprompts:] # attention of prompt vec (dim0) to embeds (dim1)
    atts = atts.detach().cpu().numpy()

    vocab = _get_vocab(tokenizer)
    for att in atts[:model.prompts.shape[0]]:
        _decode_one(vocab, att)
    print("CLS prompt:")
    for att in atts[model.prompts.shape[0]:]:
        _decode_one(vocab, att)
    print()


def bert_find_cls_prompt_interactions(model, tokenizer, layernorm=True):
    print(f"CLS (Q) - EMBED + prompts (K) scaled dot-product attention (layernorm={layernorm})")
    mprompts = model.prompts
    mprompts = mprompts.reshape((1,) + mprompts.shape)

    membs = model.model.base_model.get_input_embeddings().weight
    membs = membs.reshape((1,) + membs.shape)

    defcls = membs[:, tokenizer.cls_token_id, :].reshape(1,1,-1)
    defsep = membs[:, tokenizer.sep_token_id, :].reshape(1,1,-1)

    if model.clsprompt is not None:
        mcls = model.clsprompt.reshape(1,1,-1)
        prompts = torch.cat((defcls, defsep, mcls), dim=1)
    else:
        prompts = torch.cat((defcls, defsep), dim=1)
    embs = torch.cat((membs, mprompts), dim=1)

    if layernorm:
        tmp = model.model.base_model.embeddings.LayerNorm(torch.cat((prompts, embs), dim=1))
        prompts = tmp[:, :prompts.shape[1]]
        embs = tmp[:, prompts.shape[1]:]

    # destroy model a bit :)
    enclayeratt = model.model.base_model.encoder.layer[0].attention.self     # only first layer attention interests us
    edtatt = EditedBertSelfAttention(enclayeratt.num_attention_heads, enclayeratt.attention_head_size, enclayeratt.query, enclayeratt.key)
    edtatt.train(False)

    atts = edtatt(prompts, embs) # attentions of first layer per each head
    atts = torch.max(atts, dim=1).values    # max value from all heads
    atts = atts[0]  # ignore batch
    # atts = atts[:noprompts, noprompts:] # attention of prompt vec (dim0) to embeds (dim1)
    atts = atts.detach().cpu().numpy()

    vocab = _get_vocab(tokenizer, asnp=False)
    vocab.extend([f"***PROMPT_{i}***" for i in range(len(model.prompts))])
    vocab = np.asarray(vocab)
    for att in atts:
        _decode_one(vocab, att)



def _get_vocab(tokenizer, asnp=True):
    tv = tokenizer.vocab
    vocab = sorted(tv, key=tv.get)
    for k, v in tv.items():
        assert vocab[v] == k
    if asnp:
        return np.asarray(vocab)
    else:
        return vocab


def _decode_one(vocab: np.ndarray, sim: np.ndarray, numwords=10, maxval=True):
    assert len(sim.shape) == 1
    inds = np.argsort(sim)
    if maxval:
        inds = np.flip(inds)
    inds = inds[:numwords]  # numwords best
    txt = ',\t'.join([f"{vocab[i]:10s} ({sim[i]:>0.5f})" for i in inds])
    print(txt)


def _find_cos_sim(prompt, embedds):
    if len(prompt.shape) != 2:
        prompt = prompt.reshape((1, -1))

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cs = cos(prompt, embedds)
    return cs.numpy()


def find_cos_sim_embeddings(model, tokenizer):
    print("COSINE SIMILARITY")
    vocab = _get_vocab(tokenizer)
    embs = model.model.get_input_embeddings().weight.detach().cpu()
    prompts = model.prompts.detach().cpu()
    for p in prompts:
        sim = _find_cos_sim(p, embs)
        _decode_one(vocab, sim)
    if model.clsprompt is not None:
        print("CLS prompt:")
        sim = _find_cos_sim(model.clsprompt.detach().cpu(), embs)
        _decode_one(vocab, sim)
    print()


def find_dot_prod_embeddings(model, tokenizer):
    print("DOT PRODUCT")
    vocab = _get_vocab(tokenizer)
    embs = model.model.get_input_embeddings().weight.detach().cpu().numpy()
    prompts = model.prompts.detach().cpu().numpy()
    for p in prompts:
        sim = np.dot(embs, p)
        _decode_one(vocab, sim)
    if model.clsprompt is not None:
        print("CLS prompt:")
        sim = np.dot(embs, model.clsprompt.detach().cpu().numpy().flatten())
        _decode_one(vocab, sim)
    print()


def compare_prompts(qprompts, kprompts):
    qprompts = qprompts.detach().cpu()
    kprompts = kprompts.detach().cpu()

    vocab = [str(i) for i in range(len(kprompts))]
    for p in qprompts:
        sim = _find_cos_sim(p, kprompts)
        _decode_one(vocab, sim)


if __name__ == '__main__':
    from transformers import BertTokenizerFast, GPT2TokenizerFast
    from models.BertSoftPrompt import BertSoftPromptModel, BertSoftPromptNoCLSModel
    import paths

    # print("BERT")
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model = torch.load(paths.bin + "bert_10_rquet_fulltrain_adamW_lr0.0002_batch64/model_273_train_0.340478_test_0.544395.pth", map_location="cpu")
    #
    # print("BERT textinit")
    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model = BertSoftPromptModel("bert-base-cased", prompts=8, device="cpu")
    # model.load_learnable_params(paths.bin + "RquetDataset/bert_textinit_val_prompts8_inp119_lr0.02_batch64/" + "model_80.cp")
    #
    # print("BERT")
    # tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
    # model = torch.load(paths.bin + "bertlarge_10_rquet_fulltrain_adamW_lr0.1_batch64/model_221_train_0.513188_test_0.595530.pth", map_location="cpu")

    # print("BERT")
    # tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
    # model = torch.load(paths.bin + "bertlarge_10_rquet_fulltrain_adamW_lr0.001_batch64/model_63_train_0.365147_test_0.516654.pth", map_location="cpu")
    #
    # find_cos_sim_embeddings(model, tokenizer)
    # find_dot_prod_embeddings(model, tokenizer)
    # bert_find_similiar_embeddings(model, tokenizer)
    # bert_find_similiar_embeddings(model, tokenizer, layernorm=False)
    #

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    # model = BertSoftPromptModel("bert-base-cased", prompts=8, device="cpu")
    #
    # model.load_learnable_params(paths.bin + "RquetDataset/bert_textinit_prompts8_inp119_lr0.02_batch64/" + "model_best_val_acc.cp")
    # kprompts = model.prompts.detach().clone()
    # find_cos_sim_embeddings(model, tokenizer)
    # find_dot_prod_embeddings(model, tokenizer)
    # bert_find_similiar_embeddings(model, tokenizer)
    # bert_find_similiar_embeddings(model, tokenizer, layernorm=False)
    #
    # print()
    # print()

    model = BertSoftPromptNoCLSModel("bert-base-cased", prompts=8, device="cpu")
    model.load_learnable_params(paths.bin + "RquetDataset/bertnoCLS_textinit_prompts8_inp119_lr0.0002_batch64/" + "model_best_val_acc.cp")
    kprompts = model.prompts.detach().clone()
    compare_prompts(model.prompts, kprompts)
    find_cos_sim_embeddings(model, tokenizer)
    find_dot_prod_embeddings(model, tokenizer)
    bert_find_similiar_embeddings(model, tokenizer)
    bert_find_similiar_embeddings(model, tokenizer, layernorm=False)

    # bert_find_cls_prompt_interactions(model, tokenizer)
    # bert_find_cls_prompt_interactions(model, tokenizer, layernorm=False)
    #
    # model.load_learnable_params(
    #     paths.bin + "RquetDataset/bertnoCLS_textinit_prompts8_inp119_lr0.002_batch64/" + "model_best_val_acc.cp")
    # bert_find_cls_prompt_interactions(model, tokenizer)
    # bert_find_cls_prompt_interactions(model, tokenizer, layernorm=False)
    #
    # model.load_learnable_params(
    #     paths.bin + "RquetDataset/bertnoCLS_textinit_prompts8_inp119_lr0.02_batch64/" + "model_best_val_acc.cp")
    # bert_find_cls_prompt_interactions(model, tokenizer)
    # bert_find_cls_prompt_interactions(model, tokenizer, layernorm=False)
    #
    # # model.load_learnable_params(paths.bin + "RquetDataset/bertnoCLS_textinit_prompts8_inp119_lr0.0002_batch64/" + "model_best_val_acc.cp")
    # # compare_prompts(model.prompts, kprompts)
    # #
    # # find_cos_sim_embeddings(model, tokenizer)
    # # find_dot_prod_embeddings(model, tokenizer)
    # # bert_find_similiar_embeddings(model, tokenizer)
    # # bert_find_similiar_embeddings(model, tokenizer, layernorm=False)
    #
    # print()
    # model = BertSoftPromptNoCLSModel("bert-base-cased", prompts=8, device="cpu")
    # model.load_learnable_params(paths.bin + "RquetDataset/bert_textinit_prompts8_inp119_lr0.02_batch64/" + "model_best_val_acc.cp")
    # bert_find_cls_prompt_interactions(model, tokenizer)
    # bert_find_cls_prompt_interactions(model, tokenizer, layernorm=False)
    #
    # model.load_learnable_params(
    #     paths.bin + "old_rguet_noValSplit/bert_textinit_inp119_lr0.0002_batch64_params7681/" + "model_88_val_acc_0.783333.cp")
    # bert_find_cls_prompt_interactions(model, tokenizer)
    # bert_find_cls_prompt_interactions(model, tokenizer, layernorm=False)