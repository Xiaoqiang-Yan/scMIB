import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import random
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
import torch.distributions.normal as normal
import torch.nn.functional as F

from datasets import Loader, apply_noise
from model import AutoEncoder
from evaluate import evaluate
from util import AverageMeter
from datetime import datetime as dt
from torch.distributions import Categorical, MixtureSameFamily, Normal

def make_dir(directory_path, new_folder_name):
    """Creates an expected directory if it does not exist"""
    directory_path = os.path.join(directory_path, new_folder_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

def inference(net, data_loader_test):
    net.eval()
    feature_vector = []
    labels_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(data_loader_test):
            feature_vector.extend(net.feature(x.cuda()).detach().cpu().numpy())
            labels_vector.extend(y.numpy())
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    '''
        arg1(adata)[AnnData matrix]
        arg2(fixed_clus_count)[int]

        return:
            resolution[int]
    '''
    dis = []
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)
    i = 0
    res_new = []
    for res in resolutions:
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(
            adata.obs['leiden']).leiden.unique())
        dis.append(abs(count_unique_leiden-fixed_clus_count))
        res_new.append(res)
        if count_unique_leiden == fixed_clus_count:
            break
    reso = resolutions[np.argmin(dis)]

    return reso


def train(args):
    data_load = Loader(args, dataset_name=args["dataset"], drop_last=True)
    data_loader = data_load.train_loader
    data_loader_test = data_load.test_loader
    x_shape = args["data_dim"]

    prior_loc = torch.zeros(args["batch_size"], 128)
    prior_scale = torch.ones(args["batch_size"], 128)
    prior = normal.Normal(prior_loc, prior_scale)

    results = []

    # Hyper-params
    init_lr = args["learning_rate"]
    max_epochs = args["epochs"]
    mask_probas = [0.4]*x_shape

    # setup model
    model = AutoEncoder(
        num_genes=x_shape,
        hidden_size=128,
        masked_data_weight=0.75,
        mask_loss_weight=0.7,
        class_num=args['n_classes'],
        cluster_parameter=args["cluster"]
    ).cuda()
    # model_checkpoint = 'model_checkpoint.pth'

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    log = []
    # train model
    for epoch in range(max_epochs):
        model.train()
        meter = AverageMeter()
        for i, (x, y) in enumerate(data_loader):
            loss_list = []
            x = x.cuda()
            x_corrputed, mask = apply_noise(x, mask_probas)
            x_corrputed_latent, clean_latent, reconstruction1_loss, mask_loss, contrastive_loss = model.loss_mask(x_corrputed, x, mask)
            reconstruction_loss_val = reconstruction1_loss
            loss_list.append(reconstruction_loss_val)
            mask_loss_val = mask_loss
            loss_list.append(mask_loss_val)
            loss_list.append(contrastive_loss)

            z_corrupted = x_corrputed_latent.cpu()
            z_corrupted = F.log_softmax(torch.clamp(z_corrupted, min=1e-8), dim=-1)
            prior_sample = prior.sample()
            prior_sample = F.softmax(torch.clamp(prior_sample, min=1e-8), dim=-1)
            skl_loss = torch.nn.functional.kl_div(z_corrupted, prior_sample, reduction='batchmean')
            skl_corrupted_val = args["IB"] * skl_loss
            loss_list.append(skl_corrupted_val)

            loss = sum(loss_list)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            meter.update(loss.detach().cpu().numpy())
    
        if epoch == 80:
            # Generator in eval mode
            latent, true_label = inference(model, data_loader_test)
            if latent.shape[0] < 10000:
                clustering_model = KMeans(n_clusters=args["n_classes"])
                clustering_model.fit(latent)
                pred_label = clustering_model.labels_
            else:
                adata = sc.AnnData(latent)
                sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
                # sc.tl.umap(adata)
                reso = res_search_fixed_clus(adata, args["n_classes"])
                sc.tl.leiden(adata, resolution=reso)
                pred = adata.obs['leiden'].to_list()
                pred_label = [int(x) for x in pred]
            

            nmi, ari, acc = evaluate(true_label, pred_label)

            res = {}
            res["nmi"] = nmi
            res["ari"] = ari
            res["acc"] = acc
            res["dataset"] = args["dataset"]
            res["epoch"] = epoch
            results.append(res)

            print("\tEvalute: [nmi: %f] [ari: %f] [acc: %f]" % (nmi, ari, acc))

    # torch.save({
    #     "optimizer": optimizer.state_dict(),
    #     "model": model.state_dict()
    # }, model_checkpoint
    # )

    return results

if __name__ == "__main__":
    args = {}
    args["num_workers"] = 4
    args["paths"] = {"data": "data/",
                     "results": "./res/"}
    args['batch_size'] = 256
    args["data_dim"] = 1000
    args['n_classes'] = 6
    args['epochs'] = 81
    args["dataset"] = "Pollen"
    args["learning_rate"] = 1e-3
    args["latent_dim"] = 32
    args["IB"] = 0.5
    args["cluster"] = 0.2

    print(args)

    path = args["paths"]["data"]
    files = ["Pollen"]

    results = pd.DataFrame()
    for dataset in files:
        seed = random.randint(1, 100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        if dataset == "Pollen":
            args["n_classes"] = 11
        elif dataset == "Plasschaert":
            args["n_classes"] = 8

        print(f">> {dataset}")
        args["dataset"] = dataset
        # args["save_path"] = make_dir("/data/sc_data/scMIB/", dataset)
        # os.makedirs(args["save_path"], exist_ok=True)

        res = train(args)

        with open('res/results.txt', 'a+') as f:
                f.write(
                    'dataset:{} \t nmi:{:.4f} \t ari:{:.4f} \t acc:{:.4f} \t {} \n'.format(
                        res[0]["dataset"], res[0]["nmi"], res[0]["ari"], res[0]["acc"], dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ))
                f.flush()
