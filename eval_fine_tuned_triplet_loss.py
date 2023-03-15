import math

import numpy as np
from sklearn.metrics import accuracy_score
import paths
from dataset.swda_rq_dataset import SwDA_RQDataset_Balanced
from models.BertSoftPrompt import BertSoftPromptTripletCLSModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from models.utils.model_utils import predict_ds


ds_eval_part = "test"    # or "test
run_times = 1
few_shots = [1, 5, 10, 25]

modelid = "bert-base-cased"
input_length = 118
prompts = 10
model = BertSoftPromptTripletCLSModel(modelid, prompts=prompts, out_dim=32)

model.load_learnable_params(paths.bin + "TripletMultiDataset/bert-base-cased_out32_margin0.5_sc1_prompts10_inp118_lr0.0002_batch64/model_best_val_acc.cp")


model.eval()

trainds = SwDA_RQDataset_Balanced("train", modelid, input_length=input_length, left_context=False, right_context=False)
testds = SwDA_RQDataset_Balanced(ds_eval_part, modelid, input_length=input_length, left_context=False, right_context=False)

results = []

base_model_path = paths.bin + "TripletMultiDataset/bert-base-cased_out32_margin0.5_mrda_rquet_frozen_head_prompts10_inp118_lr0.0002_batch32/model_best_val_acc.cp"
model.load_learnable_params(base_model_path)

testx, testy = predict_ds(testds, model)
print(f"Totally {len(testx)} samples from testds")



for few_shot in few_shots:
    res = []
    for run in range(run_times):
        #current_model_path = base_model_path.replace("shotsN", "shots"+str(few_shot)).replace("seedM", "seed"+str(run))
        #model.load_learnable_params(current_model_path)

        print(f"************* {few_shot}-shot run {run} *************")
        resitem = []
        trainds.few_shot(few_shot, seed=run)

        trainx, trainy = predict_ds(trainds, model)
        print(f"Totally {len(trainx)} samples from trainds")

        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"1-NN cos acc: {acc}")

        knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"1-NN euc acc: {acc}")

        knn = KNeighborsClassifier(n_neighbors=1, metric="manhattan")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"1-NN manhattan acc: {acc}")

        shots = len(trainx)//2
        n=int(math.sqrt(shots))

        knn = KNeighborsClassifier(n_neighbors=n, metric="cosine")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"{n}-NN cos acc: {acc}")

        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"{n}-NN euc acc: {acc}")

        knn = KNeighborsClassifier(n_neighbors=n, metric="manhattan")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"{n}-NN manhattan acc: {acc}")


        knn = SVC()
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"SVC acc: {acc}")


        knn = SVC(kernel="linear")
        knn.fit(trainx, trainy)
        predy = knn.predict(testx)
        acc = accuracy_score(testy, predy)
        resitem.append(acc)
        print(f"SVC linear acc: {acc}")

        res.append(resitem)
    results.append(res)



print("#######################################")
meanacc = np.mean(results, axis=1)
stdacc = np.std(results, axis=1)
confint = 1.96 * stdacc / math.sqrt(run_times)

print(f"mean ACC +- conf interval {run_times} runs *****************")
print("1-NN\t1-NN euc\t1-NN mhtn\tn-NN\tn-NN euc\tn-NN mhtn\tSVC\tSVC linear")
for i in range(len(meanacc)):
    a = meanacc[i]
    ci = confint[i]
    print("\t".join([f"{a[x]} +- {ci[x]}" for x in range(len(a))]))

print()
print(f"mean ACC {run_times} runs *****************")
print("1-NN\t1-NN euc\t1-NN mhtn\tn-NN\tn-NN euc\tn-NN mhtn\tSVC\tSVC linear")
for i in range(len(meanacc)):
    print("\t".join([str(x) for x in meanacc[i]]))

print()
print(f"conf interval {run_times} runs *****************")
print("1-NN\t1-NN euc\t1-NN mhtn\tn-NN\tn-NN euc\tn-NN mhtn\tSVC\tSVC linear")
for i in range(len(meanacc)):
    print("\t".join([str(x) for x in confint[i]]))


#
#
# with open(f"{model.__class__.__name__}_cls_vectors.csv", mode="w", encoding="utf8") as fw1:
#     for v in valx:
#         fw1.write("\t".join([str(tmp) for tmp in v]) + "\n")
#
# with open(f"{model.__class__.__name__}_labels.csv", mode="w", encoding="utf8") as fw2:
#     for v in valy:
#         fw2.write(f"{v}\n")
#
# with open(f"triplet_test_cls_vectors.csv", mode="w", encoding="utf8") as fw1:
#     for v in testx:
#         fw1.write("\t".join([str(tmp) for tmp in v]) + "\n")
#
# with open(f"triplet_test_labels.csv", mode="w", encoding="utf8") as fw2:
#     for v in testy:
#         fw2.write(f"{v}\n")



# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
#
# def vis(x, y, lr="auto", perpl=20, metric="euclidean"):
#     x = np.asarray(x)
#     xemb = TSNE(n_components=2, learning_rate=lr, perplexity=perpl, metric=metric).fit_transform(x)
#
#     plt.figure()
#     plt.scatter(xemb[:,0], xemb[:,1], c=y)
#     plt.title(f"T-SNE(perplexity={perpl}, metric={metric})")
#
#
# def vispca(x, y):
#     x = np.asarray(x)
#     pca = PCA(n_components=2)
#     pca.fit(x)
#     xpca = pca.transform(x)
#
#     plt.figure()
#     plt.scatter(xpca[:, 0], xpca[:, 1], c=y)
#     plt.title(f"PCA (exp. var = {sum(pca.explained_variance_ratio_)}; {pca.explained_variance_ratio_})")
#
#
# vispca(testx, testy)
# vis(testx, testy, metric="cosine")
# vis(testx, testy, metric="euclidean")
# vis(testx, testy, metric="manhattan")
#
# plt.show(block=True)
#
