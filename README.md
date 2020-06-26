# Graph Neural Networks for the pT1 Gland Graph Dataset
This repo contains the medical dataset, different Graph Neural Network implementations, the python code, the classification results and the Hyperparameters used
for the paper entitled as "Classification of Intestinal Gland Cell-Graphs using Graph Neural Networks" and submitted at the International Conference of Pattern Recognition (ICPR) January 2021.

This repo contains different Graph Neural Network (GNN) implementations for Graph classification.
The models are applied on the publicly available [pT1 Gland Graph Dataset (GG-pT1)](https://github.com/LindaSt/pT1-Gland-Graph-Dataset).

(In this repo the term "paper" correspond to the 4 node feature set and the term "base" corresponds to the 33 node feature set used for the paper submitted at the ICPR)

The Hyperparameters are found [here](https://github.com/waljan/GNNpT1/tree/master/Hyperparameters).

## Models:
So far, the following models are implemented:

- **GCN**: uses the Graph convolution operator introduced by [Kipf & Welling 2016](https://arxiv.org/abs/1609.02907).
    - **GCNWithJK**: uses the JumpingKnowledge layer aggregation module introduced by [Xu et al. 2018](https://arxiv.org/abs/1806.03536)

- **GraphSAGE**: uses the GraphSAGE operator with mean-aggregator introduced by [Hamilton et al. 2017](https://arxiv.org/abs/1706.02216)
    - **GraphSAGEWithJK**: uses the JumpingKnowledge layer aggregation module introduced by [Xu et al.](https://arxiv.org/abs/1806.03536)

- **GAT**: uses a attention-based message passing layer intorduced by [Velickovic et al. 2017](https://arxiv.org/abs/1710. In this repo it is called GATNet.

- **enn**: Edge Network from [Gilmer et al. 2017](https://arxiv.org/abs/1704.01212). It uses edge features to update the hidden representation of nodes. In this repo it is called NMP.

- **GIN**: Graph Isomorphism Network from [Xu et al. 2018](https://arxiv.org/abs/1810.00826) with epsilon=0.

- **1-GNN**: one of the Graph Neural Network baselines used in [Morris et al. 2019](https://arxiv.org/abs/1810.02244). In this repo it is called GraphNN.






## Dataset: pT1 Gland Graph Dataset (GG-pT1)
[This repo](https://github.com/LindaSt/pT1-Gland-Graph-Dataset) contains the intestinal gland segmentation dataset from pT1 cancer patients.
It includes:

- **Dataset**: From each image there are 26 images of cropped glands (13 normal, 13 dysplastic). 
  - image_labels.csv: Classification label for each graph and image (normal or dysplastic)
  - There is a folder for each image. In this folder there is a folder for each crop containing:
    - Cropped out gland image (*-image.jpg)
    - Annotation mask (*-gt.png)
    - Excel file with the features (*-features.xlsx)



- **Text files**: 
  - dataset_split.csv: reference, validation and test set split for all 4 cross-validations
  - feature_overview.csv: list of all possible node features (with enumeration, mean and std)
  - ged-costs.csv: parameters for the different experiments



- **Graphs**:
  - Base dataset: cell segmentation with features (just nodes)
  - Paper graphs:
    - Baseline
    - Optimized graph



The dataset is submitted for publishing at the [COMPAY19 Workshop](https://openreview.net/group?id=MICCAI.org/2019/Workshop/COMPAY) ([link to the paper](https://openreview.net/pdf?id=HklExX79-S)).
The parameters for the GED calculated in this paper can be found [here](https://bit.ly/2xDuRcV).


If you want to cite us please use:
`` BIBTEX TBA``

This work is part of a larger project. Find out more [here](https://icosys.ch/bts-project).



## Requirements:
To use this repository you can create a virtual environment from the requirement.yml file:
```
conda env create -f requirements.yml
```


To remove the conda environment run the following code:
```
# first deactivate the conda environment that you want to remove
conda deactivate

# remove the GNNEnv environment
conda env remove -n GNNEnv
```
