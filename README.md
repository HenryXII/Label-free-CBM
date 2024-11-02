# Using Automatic Concept Correction to improve LF-CBM

This is the repository for my DSC180A project where I used the technique of Automatic Concept Correction to further improve LF-CBM (https://github.com/Trustworthy-ML-Lab/Label-free-CBM). 

## Setup

1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Download pretrained models by running  `bash download_models.sh` (they will be unpacked to `saved_models`)
4. Download and process CUB dataset by running `bash download_cub.sh` 
5. Download ResNet18(Places365) backbone by running `bash download_rn18_places.sh`

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet you must set the correct path in `DATASET_ROOTS["imagenet_train"]` and `DATASET_ROOTS["imagenet_val"]` variables in `data_utils.py`.

## Running the models

### 1. Train LF-CBM

Train a concept bottleneck model on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`

### 2. Reproduce ACC results.

Train a concept bottleneck model with ACC on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt --acc data/concept_sets/merged_cifar10_concepts.json`

### 3. Evaluate trained models

Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

Additional evaluations and reproductions of our model editing experiments are available in the notebooks of `experiments` directory.




