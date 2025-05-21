# scMIB

scMIB: Compact and Informative Representation Learning for scRNA-seq Data Clustering

## Installation

Clone this repository. The stAA has been implemented in Python3.9.16 and Pytorch 1.12.0.

```
git clone https://github.com/Xiaoqiang-Yan/scMIB.git
cd scMIB-main/
```

## Datasets

h5 file contains gene expression X and true label Y.

## Usage

We provided some demos to demonstrate usage of scMIB.

```Python
# args["dataset"].h5 save in args["paths"]["data"] directory (h5 file contains gene expression X and true label Y)
# hyperparameter
args = {}
args["dataset"] = "Pollen" # dataset name
args['n_classes'] = 6 # number of clusters
args["paths"] = {"data": "/data/", "results": "/res/results.txt"} # Datasets directory and output directory
args['batch_size'] = 256 
args["data_dim"] = 1000 # num_features of high variable genes
args['epochs'] = 81
args["num_workers"] = 4
args["learning_rate"] = 1e-3 
args["latent_dim"] = 32 # latent embedding dim
train(args)
```
