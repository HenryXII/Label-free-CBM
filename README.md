## Setup

1. Install Python (3.9) and PyTorch (1.13).
2. Install dependencies by running `pip install -r requirements.txt`
3. Download pretrained models by running  `bash download_models.sh` (they will be unpacked to `saved_models`)
4. Download and process CUB dataset by running `bash download_cub.sh` 
5. Download ResNet18(Places365) backbone by running `bash download_rn18_places.sh`

We do not provide download instructions for ImageNet data, to evaluate using your own copy of ImageNet you must set the correct path in `DATASET_ROOTS["imagenet_train"]` and `DATASET_ROOTS["imagenet_val"]` variables in `data_utils.py`.

## Running the models

### 1. Creating Concept Sets (Optional):
A. Create initial concept set using GPT-3 - `GPT_initial_concepts.ipynb`, do this for all 3 prompt types (can be skipped if using the concept sets we have provided). NOTE: This step costs money and you will have to provide your own `openai.api_key`.

B. Process and filter the conceptset by running `GPT_conceptset_processor.ipynb` (Alternatively get ConceptNet concepts by running ConceptNet_conceptset.ipynb)

### 2. Train LF-CBM

Train a concept bottleneck model on CIFAR10 by running:

`python train_cbm.py --concept_set data/concept_sets/cifar10_filtered.txt`


### 3. Evaluate trained models

Evaluate the trained models by running `evaluate_cbm.ipynb`. This measures model accuracy, creates barplots explaining individual decisions and prints final layer weights which are the basis for creating weight visualizations.

Additional evaluations and reproductions of our model editing experiments are available in the notebooks of `experiments` directory.

### 4. Reproduce ACC results.



