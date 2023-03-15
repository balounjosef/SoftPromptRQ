import numpy as np
import torch
from torch.utils.data import DataLoader


def freeze_model(model):
    print("Freezing model")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        assert param.requires_grad is False, f"{name} requires grad {param.requires_grad}"


def decide_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return device


def prompts_norms(prompts):
    norms = torch.linalg.vector_norm(prompts, ord=2, dim=-1)
    norms = norms.detach().cpu().numpy()
    return norms


def test_overfit(model, tokenizer, lr=0.02, ep=20, modeltrain=False):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable (requires_grad is {param.requires_grad})")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id     # does not matter - it is masked
    x = tokenizer(["false", "true"], padding="max_length", truncation=True, max_length=5, return_tensors="pt")

    # x.input_ids = x.input_ids.to(model.device)
    # x.attention_mask = x.attention_mask.to(model.device)

    # checking of frozen weights, initial weights and predictions
    model.eval()
    if str(model.model).startswith("T5Model"):
        transformer_out = model.model(input_ids=x.input_ids.to(model.device), attention_mask=x.attention_mask.to(model.device), decoder_input_ids=torch.tensor([[0], [0]]).to(model.device)).last_hidden_state
    else:
        transformer_out = model.model(input_ids=x.input_ids.to(model.device), attention_mask=x.attention_mask.to(model.device)).last_hidden_state
    model_out = model(x.input_ids, x.attention_mask)
    prompts_init = model.prompts.clone()
    print(f"init prediction: {model_out}")

    # few weight updates
    model.train(modeltrain)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    y = torch.tensor([[0.], [1.]]).to(model.device)
    for i in range(ep):
        # Compute prediction and loss
        pred = model(x.input_ids, x.attention_mask).sum(dim=-1, keepdim=True)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{i:>5}] {loss.item():>5f} \t preds: {pred[0].item()}, {pred[1].item()}")

    # compare the weights with initialized one
    model.eval()
    if str(model.model).startswith("T5Model"):
        transformer_out_2 = model.model(input_ids=x.input_ids.to(model.device), attention_mask=x.attention_mask.to(model.device), decoder_input_ids=torch.tensor([[0], [0]]).to(model.device)).last_hidden_state
    else:
        transformer_out_2 = model.model(input_ids=x.input_ids.to(model.device), attention_mask=x.attention_mask.to(model.device)).last_hidden_state
    model_out_2 = model(x.input_ids, x.attention_mask)
    print(f"trained prediction: {model_out_2}")

    transformer_diff = torch.sum(torch.abs(transformer_out - transformer_out_2))
    model_diff = torch.sum(torch.abs(model_out - model_out_2))
    prompts_diff = torch.sum(torch.abs(prompts_init - model.prompts))

    print(f"transformer_diff {transformer_diff}")
    print(f"model_diff {model_diff}")
    print(f"prompts_diff {prompts_diff}")

    # assert transformer_diff < 0.0000000001, "The frozen transformer model prediction should be the same after weight update"
    assert model_diff > 0.01, "The whole model prediction should be different due to updated prompts and classifier head"
    assert prompts_diff > 0.01, "Prompts should be modified"
    assert model_out_2[0].sum().item() < 0 and model_out_2[1].sum().item() > 0, "Model not trained"


def decide_tokenizer(tokenizerid):
    if "gpt" in tokenizerid:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizerid, truncation_side='left', padding_side='left')
        sep_text = tokenizer.eos_token  # "<|endoftext|>"
        last_token = tokenizer.eos_token
    elif "bloom" in tokenizerid:
        from transformers import BloomTokenizerFast
        tokenizer = BloomTokenizerFast.from_pretrained(tokenizerid, truncation_side='left', padding_side='left')
        sep_text = tokenizer.eos_token  # '</s>'
        last_token = tokenizer.eos_token
    elif "roberta" in tokenizerid:
        from transformers import RobertaTokenizerFast
        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizerid, truncation_side='left')
        sep_text = tokenizer.sep_token  # '</s>'
        last_token = ""    # adds '[CLS] text [SEP]' by default
    elif "bert" in tokenizerid:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(tokenizerid, truncation_side='left')
        sep_text = tokenizer.sep_token  # '[SEP]'
        last_token = ""    # adds '[CLS] text [SEP]' by default
    elif "t5" in tokenizerid:
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained(tokenizerid, truncation_side='left')
        sep_text = tokenizer.eos_token  # '</s>'
        last_token = ""    # adds 'text [SEP]' by default
    else:
        raise NotImplementedError(f"Do not know which tokenizer class for {tokenizerid}")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id  # does not matter - it is masked
    # tokenizer instance, separator text, last token

    print(f"Tokenizer decided: {tokenizer.__class__.__name__}")
    return tokenizer, sep_text, last_token


def predict_ds(ds, model, batch_size=64, asnumpy=False):
    resx = []
    resy = []

    model.eval()

    for x, xm, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
        preds = model(x, xm)
        preds = preds.detach().cpu().tolist()
        lbls = y.flatten().long().detach().cpu().tolist()
        resx.extend(preds)
        resy.extend(lbls)

    if asnumpy:
        return np.asarray(resx), np.asarray(resy)
    else:
        return resx, resy


if __name__ == '__main__':
    decide_tokenizer("gpt2")
    decide_tokenizer("bigscience/bloom-560m")
    decide_tokenizer("roberta-base")
    decide_tokenizer("bert-base-cased")
    decide_tokenizer("google/t5-v1_1-small")
