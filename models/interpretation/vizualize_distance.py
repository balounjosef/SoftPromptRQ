import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import paths
from dataset.mrda_dataset import MRDADataset
from dataset.rquet_dataset import RquetDataset
from dataset.sarcasm_v2_dataset import SarcasmV2Dataset
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from models.utils.model_utils import predict_ds


def _get_dists(ax, ay, dsx, dsy):
    assert len(ax) == len(ay) == 2, "only 1 shot"

    euc = distance.cdist(ax, dsx, 'euclidean')
    cos = distance.cdist(ax, dsx, 'cosine')

    reseuc = {
        f"{ay[0]}pos": euc[0][dsy == ay[0]],
        f"{ay[0]}neg": euc[0][dsy != ay[0]],
        f"{ay[1]}pos": euc[1][dsy == ay[1]],
        f"{ay[1]}neg": euc[1][dsy != ay[1]],
    }
    rescos = {
        f"{ay[0]}pos": cos[0][dsy == ay[0]],
        f"{ay[0]}neg": cos[0][dsy != ay[0]],
        f"{ay[1]}pos": cos[1][dsy == ay[1]],
        f"{ay[1]}neg": cos[1][dsy != ay[1]],
    }
    return reseuc, rescos


def _plot(res, title="no title"):
    pos = 1
    plt.figure(figsize=[12., 4.8])
    for dsk, r in res.items():
        plt.boxplot([r["0pos"], r["1pos"], r["0neg"], r["1neg"]], positions=[pos, pos + 1, pos + 2, pos + 3], labels=[f"{dsk} 0pos", f"{dsk} 1pos", f"{dsk} 0neg", f"{dsk} 1neg"])
        pos += 5
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()


if __name__ == '__main__':
    run_times = 10

    modelid = "bert-base-cased"
    input_length = 118
    prompts = 10
    model = BertSoftPromptTripletCLSModel(modelid, prompts=prompts, out_dim=32)

    model.load_learnable_params(paths.bin + "TripletMultiDataset/bert-base-cased_out32_margin0.5_sc1_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")
    model.eval()

    ads = SwDA_RQDataset_Balanced("train", modelid, input_length=input_length, left_context=False, right_context=False)

    p = "train"
    dsxy = {
        "MRDA": predict_ds(MRDADataset(p, modelid, input_length=input_length), model, asnumpy=True),
        "SarcasmV2": predict_ds(SarcasmV2Dataset(p, modelid, input_length=input_length), model, asnumpy=True),
        "Rquet": predict_ds(RquetDataset(p, modelid, input_length=input_length, left_context=False, right_context=False), model, asnumpy=True)
    }

    reseuc = {
        "MRDA": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        },
        "SarcasmV2": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        },
        "Rquet": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        }
    }
    rescos = {
        "MRDA": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        },
        "SarcasmV2": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        },
        "Rquet": {
            "0pos": [],
            "0neg": [],
            "1pos": [],
            "1neg": []
        }
    }

    for run in range(run_times):
        print(f"************* {run} *************")
        resitem = []
        ads.few_shot(1, seed=run)

        ax, ay = predict_ds(ads, model, asnumpy=True)

        for k, (dsx, dsy) in dsxy.items():
            euc, cos = _get_dists(ax, ay, dsx, dsy)
            for k2 in euc.keys():
                reseuc[k][k2].extend(euc[k2])
                rescos[k][k2].extend(cos[k2])

    _plot(reseuc, title="euclidean")
    _plot(rescos, title="cosine")
    plt.show(block=True)


