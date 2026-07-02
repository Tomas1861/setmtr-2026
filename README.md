# SetMTr: Predicting the Validity of Set Data with a Self-supervised Masked Transformer

## Title

SetMTr: Predicting the Validity of Set Data with a Self-supervised Masked Transformer

## Description

This repository contains the implementation and data-preparation utilities for SetMTr, a self-supervised masked Transformer model for validity prediction of set-structured data.

SetMTr treats each sample as an unordered set. During training, it masks part of a valid set and learns to reconstruct the missing elements. At inference time, reconstruction quality is used as a validity score: valid sets are expected to be reconstructed more reliably than invalid or incoherent sets.

The repository includes:

- PyTorch implementation of the SetMTr model.
- Dataset preprocessing utilities for discrete and continuous set data.
- Training and evaluation code for multiple benchmark datasets.
- Configuration files for reproducing experiments across different domains.

## Dataset Information

The experiments use set-structured datasets from several domains:

- **Chuancai and Yuecai recipes**: Chinese recipe ingredient sets from Rong et al., "Exploring Chinese Dietary Habits Using Recipes Extracted From Websites", IEEE Access, DOI: https://doi.org/10.1109/ACCESS.2019.2900504.
- **Groceries**: grocery transaction sets from the Groceries benchmark dataset distributed with the R `arules` package: https://cran.r-project.org/package=arules.
- **Inorganic compound**: element-composition sets constructed from the Wikipedia list of CAS numbers by chemical compound: https://en.wikipedia.org/wiki/List_of_CAS_numbers_by_chemical_compound.
- **Triangle**: simulated 2D point sets generated according to triangle-inequality rules.
- **SetMNIST**: 2D point sets constructed from MNIST: http://yann.lecun.com/exdb/mnist/.
- **ModelNet10**: 3D point sets sampled from Princeton ModelNet10 meshes: https://modelnet.cs.princeton.edu/.
- **Metabolic reactions**: reaction-metabolite sets from BiGG Models, including iAF1260b, iHN637, iIT341, and iJO1366: https://bigg.ucsd.edu/.

Dataset files are organized as follows:

- `datasets/txts/`: line-based text files. Each line represents one set sample. Elements are separated by commas, and an optional set name may appear before a colon.
- `datasets/pkls/`: processed pickle files used by the training and evaluation scripts.
- `datasets/original_data_process/`: raw-data conversion scripts for individual datasets, when raw data is available.

Examples of the text format:

```text
Ba,I,H,O
pineapple_pork:pork,pineapple,green_pepper,red_pepper
563-63-3:Ag@[1],C@[2],H@[3],O@[2]
@[1 2],@[2 3],@[3 4]
```

Some third-party raw datasets may need to be downloaded from their original providers before full preprocessing can be repeated.

## Code Information

Important files and directories:

- `setmtr/main.py`: main entry point for training and evaluation.
- `setmtr/config.yaml`: dataset paths, task selection, and hyperparameter settings.
- `setmtr/models.py`: SetMTr model definitions.
- `setmtr/modules.py`: neural network modules used by the model.
- `setmtr/criteria.py`: loss functions and evaluation metrics.
- `setmtr/dataset.py`: dataset and dataloader utilities.
- `setmtr/utils.py`: helper functions.
- `datasets/preprocess.py`: converts text-format datasets into pickle files and generates invalid examples.
- `scripts/reproduce_simulated_datasets.py`: reproduces the simulated Triangle dataset and invalid examples.

The implementation is based on PyTorch and PyTorch Lightning, with experiment configuration managed through `setmtr/config.yaml`.

## Usage Instructions

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate the simulated Triangle dataset:

```bash
python scripts/reproduce_simulated_datasets.py --output-dir datasets/txts --num-real 20000 --num-fake 4000 --seed 50
```

Prepare pickle datasets from text files:

```bash
cd datasets
mkdir -p pkls
python preprocess.py
```

Train and evaluate a configured task:

```bash
cd ../setmtr
# Edit the `task` field in config.yaml, for example: task: groceries
python main.py
```

The training script reports evaluation metrics such as AUC, accuracy, F1 score, and reconstruction-related metrics for the selected dataset. To reproduce paper-level averages, run each task specified in `setmtr/config.yaml` with the corresponding dataset files and average the reported metrics across repeated splits.

## Requirements

The main dependencies are listed in `requirements.txt`:

- Python 3.9 or later
- `torch>=2.0`
- `pytorch-lightning>=2.0,<2.1`
- `torchmetrics>=0.11,<0.12`
- `torchvision>=0.15`
- `numpy>=1.23`
- `PyYAML>=6.0`
- `scikit-learn>=1.0`
- `beautifulsoup4>=4.12`
- `open3d>=0.17`
- `torchility==0.9`

MATLAB or GNU Octave is required only when repeating the `.mat` to `.txt` conversion for metabolic reaction data.

## Methodology

All datasets are converted into a common set representation. Each sample contains a collection of elements, and each element may be represented by a symbolic name, a feature vector, or both. Named elements are mapped to integer IDs, while feature-valued elements such as 2D or 3D points are stored as coordinate vectors.

Positive examples are real or rule-valid sets. Negative examples are generated by dataset-specific procedures:

- Recipe, grocery, and inorganic-compound datasets: replace part of a real set with elements sampled from the global element pool.
- Triangle: generate invalid three-point sets that violate the triangle inequality.
- SetMNIST: use non-target digits or random point sets as invalid examples.
- ModelNet10: use point clouds sampled from non-target categories.
- Metabolic reactions: use universal reactions not present in the model-specific reaction set.

SetMTr is trained using a masked reconstruction objective on positive sets. During evaluation, reconstruction quality is transformed into a validity score and compared against true valid/invalid labels.

## Citations

If this code or dataset preparation workflow is used, please cite the accompanying SetMTr manuscript.

Relevant external data sources include:

- Rong et al., "Exploring Chinese Dietary Habits Using Recipes Extracted From Websites", IEEE Access, DOI: https://doi.org/10.1109/ACCESS.2019.2900504.
- Hahsler et al., `arules`: Mining Association Rules and Frequent Itemsets, R package: https://cran.r-project.org/package=arules.
- MNIST database: http://yann.lecun.com/exdb/mnist/.
- Princeton ModelNet: https://modelnet.cs.princeton.edu/.
- BiGG Models: https://bigg.ucsd.edu/.

## License & Contribution Guidelines

No public reuse license is currently included in this review package. The code is provided for inspection and reproducibility of the submitted manuscript. Before public release, the authors should add an explicit open-source license, such as MIT, BSD-3-Clause, or Apache-2.0.

Suggested contribution workflow for future public development:

- Open an issue to discuss bugs, dataset problems, or feature requests.
- Submit focused pull requests with clear descriptions of the changes.
- Include reproduction steps and relevant configuration changes for experiment-related contributions.
- Avoid committing large raw datasets unless redistribution is permitted by the original data provider.
