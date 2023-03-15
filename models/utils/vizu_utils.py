import torch
from torch.utils.data import DataLoader


# https://projector.tensorflow.org/
def perform_triplet_vizualization(model, dataset, model_weights_filepath):
    dataloader = DataLoader(dataset, batch_size=50, shuffle=False)
    size = len(dataloader.dataset)
    positives_outputs, negatives_outputs = [], []
    # load model params
    model.load_learnable_params(model_weights_filepath)
    model.train(False)
    print("Producing CLS vectors...")
    for batch, (anchor_sample,
        anchor_attention_mask,
        anchor_label,
        positive_sample,
        positive_attention_mask,
        negative_sample,
        negative_attention_mask) in enumerate(dataloader):

        if batch % 10 == 0:
            current = batch * len(anchor_sample)
            print(f"[{current:>5d}/{size:>5d}")

        with torch.no_grad():
            # Compute prediction for anchor, positive, negative
            anchor_out = model(anchor_sample, anchor_attention_mask)
            positive_out = model(positive_sample, positive_attention_mask)
            negative_out = model(negative_sample, negative_attention_mask)

        positive_anchor_mask = anchor_label.flatten().numpy().astype(bool)

        positives_to_anchor = positive_out.cpu().detach().numpy()
        negatives_to_anchor = negative_out.cpu().detach().numpy()

        positives_outputs.extend(positives_to_anchor[positive_anchor_mask])
        negatives_outputs.extend(positives_to_anchor[~positive_anchor_mask])
        positives_outputs.extend(negatives_to_anchor[~positive_anchor_mask])
        negatives_outputs.extend(negatives_to_anchor[positive_anchor_mask])
    print("Creating csv file for visualization. See ", f"{model.__class__.__name__}_cls_vectors.csv")
    with open(f"{model.__class__.__name__}_cls_vectors.csv", mode="w", encoding="utf8") as fw1:
        with open(f"{model.__class__.__name__}_labels.csv", mode="w", encoding="utf8") as fw2:
            for positive_sample, negative_sample in zip(positives_outputs, negatives_outputs):
                row = ""
                for component in positive_sample:
                    row += str(component) + "\t"
                fw1.write(row[:-1]+"\n")
                fw2.write("POS\n")
                row = ""
                for component in negative_sample:
                    row += str(component) + "\t"
                fw1.write(row[:-1]+"\n")
                fw2.write("NEG\n")
