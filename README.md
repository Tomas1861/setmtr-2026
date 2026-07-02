# Predicting the Validity of Set Data with a Self-supervised Masked Transformer-2026

## Title

Predicting the Validity of Set Data with a Self-supervised Masked Transformer-2026

## Description

This code package contains the implementation and dataset preprocessing utilities for SetMTr, a self-supervised masked Transformer model for validity prediction of set-structured data.

SetMTr addresses the problem of deciding whether a set is valid, coherent, or likely to be real. The model is trained with a masked reconstruction objective: part of a valid input set is masked, the model reconstructs the missing elements, and reconstruction quality is used as a validity score. Valid sets are expected to be reconstructed more accurately than invalid sets because their elements follow stronger co-occurrence patterns.

The current package contains:

- `setmtr/`: model implementation, data loading, loss functions, metrics, utilities, and experiment configuration.
- `datasets/preprocess.py`: conversion from common text-format set data to pickle files used by the model.
- `datasets/readme.md`: brief description of the expected text and pickle dataset formats.
- `datasets/original_data_process/`: raw-data conversion scripts and source/intermediate files for several datasets.

## Dataset Information

The manuscript evaluates SetMTr on 11 benchmark datasets from synthetic and real-world domains:

- **Chuancai and Yuecai**: Chinese recipe ingredient sets from Rong et al., "Exploring Chinese Dietary Habits Using Recipes Extracted From Websites".
- **Groceries**: grocery transaction sets distributed with the R `arules` benchmark data collection and described in grocery basket recommendation literature.
- **Inorganic compound**: chemical element-composition sets constructed from the Wikipedia list of CAS numbers by chemical compound.
- **Triangle**: synthetic 2D point sets generated according to triangle-inequality rules.
- **SetMNIST**: 2D point sets constructed from MNIST images resized to 10 x 10, where non-zero pixels are treated as set elements.
- **ModelNet10-sofa**: 3D point clouds sampled from Princeton ModelNet10 meshes.
- **Metabolic reactions**: reaction-metabolite sets from BiGG genome-scale metabolic models, including iAF1260b, iHN637, iIT341, and iJO1366.

The code configuration also includes additional or auxiliary tasks such as `market_basket`, `market_basket_optimisation`, `modelnet10_chair`, `iAB_RBC_283`, `iAF692`, SetMNIST random-negative variants, and case-analysis tasks.

The expected processed data layout is:

- `datasets/txts/`: text-format set files. Each line represents one set sample. Elements are separated by commas, and an optional set name may appear before a colon.
- `datasets/pkls/`: pickle files generated from `datasets/txts/` by `datasets/preprocess.py`.
- `datasets/original_data_process/`: scripts and raw/intermediate files for regenerating selected text-format datasets.

Examples of the text set format:

```text
Ba,I,H,O
pineapple_pork:pork,pineapple,green_pepper,red_pepper
563-63-3:Ag@[1],C@[2],H@[3],O@[2]
@[1 2],@[2 3],@[3 4]
```

Some third-party datasets must be downloaded from their original providers before full preprocessing can be repeated. The current directory includes raw-data conversion material for SetMNIST, Triangle, inorganic compounds, and metabolic reactions, but does not include all final `datasets/txts/` and `datasets/pkls/` files.

## Code Information

Important files and directories:

- `setmtr/main.py`: main training and evaluation entry point.
- `setmtr/config.yaml`: task selection, dataset paths, and hyperparameters.
- `setmtr/models.py`: top-level SetMTr model definitions.
- `setmtr/modules.py`: Transformer-style encoder/decoder modules.
- `setmtr/criteria.py`: reconstruction losses, Hungarian matching based metrics, AUC, accuracy, and F1 metrics.
- `setmtr/dataset.py`: PyTorch dataset and dataloader utilities.
- `setmtr/utils.py`: YAML loading support, dataset loading, and helper utilities.
- `datasets/preprocess.py`: creates repeated train/test pickle datasets and negative examples.
- `datasets/original_data_process/SetMNIST/data_gen.py`: converts selected MNIST digits into 2D point-set text files.
- `datasets/original_data_process/Triangle/data_gen.py`: generates valid Triangle text-format data.
- `datasets/original_data_process/inorganic_compounds/data_gen.py`: parses inorganic compound formulas from the included HTML table.
- `datasets/original_data_process/metabolic_reactions/data_gen_mat2txt.m`: converts metabolic reaction `.mat` files into text-format sets.

The model is implemented with PyTorch and uses PyTorch Lightning/TorchMetrics-style dependencies together with the `torchility` training helper used by `setmtr/main.py`.

## Usage Instructions

Create and activate a Python environment:

```bash
cd /Users/zhou/Desktop/setmtr-main
python -m venv .venv
source .venv/bin/activate
```

Install the required Python packages. The current directory does not include a `requirements.txt` file, so install the dependencies manually:

```bash
pip install torch pytorch-lightning torchmetrics torchvision numpy PyYAML scikit-learn scipy beautifulsoup4 matplotlib open3d torchility
```

Prepare text-format datasets under `datasets/txts/`. For example, to regenerate selected source text files:

```bash
cd datasets/original_data_process/Triangle
python data_gen.py

cd ../SetMNIST
python data_gen.py

cd ../inorganic_compounds
python data_gen.py
```

Move or copy the generated `.txt` files into `datasets/txts/` using the filenames expected by `setmtr/config.yaml` and `datasets/preprocess.py`.

Create pickle datasets:

```bash
cd /Users/zhou/Desktop/setmtr-main/datasets
mkdir -p txts pkls
python preprocess.py
```

Select a task in `setmtr/config.yaml`, for example:

```yaml
task: groceries
```

Train and evaluate:

```bash
cd /Users/zhou/Desktop/setmtr-main/setmtr
python main.py
```

The script reports metrics such as AUC, accuracy, F1 score, and reconstruction-related scores for the selected task. To reproduce manuscript-level results, run the relevant tasks with the corresponding prepared data and average metrics across the repeated splits generated by `datasets/preprocess.py`.

## Requirements

Main Python dependencies inferred from the code are:

- Python 3.9 or later
- PyTorch
- PyTorch Lightning
- TorchMetrics
- TorchVision
- NumPy
- PyYAML
- SciPy
- scikit-learn
- BeautifulSoup4
- Matplotlib
- Open3D
- torchility

Additional tools:

- MATLAB or GNU Octave is needed only to rerun `datasets/original_data_process/metabolic_reactions/data_gen_mat2txt.m`.
- Internet access is needed if MNIST is downloaded automatically by TorchVision.
- ModelNet10 data must be obtained from the Princeton ModelNet source if regenerating 3D point-cloud datasets.

## Methodology

All datasets are converted into a common line-based set representation. Each line corresponds to one set instance. Elements may be represented by symbolic names, feature vectors, or both. Named elements are mapped to integer IDs, while feature-valued elements, such as 2D or 3D points, are stored as coordinate vectors. Sets with fewer than two elements are removed because they provide limited co-occurrence information.

Positive examples are real or rule-valid sets. Negative examples are generated according to dataset type:

- Recipe, grocery, and inorganic-compound datasets: replace elements in real sets with elements sampled from the global element pool.
- Triangle: replace one point so that the three points become collinear or otherwise violate valid triangle structure.
- SetMNIST: use non-target handwritten digits or random point sets as invalid examples.
- ModelNet10-sofa: use point clouds sampled from non-target object categories.
- Metabolic reactions: use non-model universal reactions supplied with the metabolic models.

SetMTr uses a Transformer-inspired encoder-decoder architecture. The encoder extracts features from masked set inputs while respecting the unordered nature of sets. The decoder reconstructs elements using element query vectors and encoder outputs. Reconstruction mismatch between the predicted set and the original set is computed with a set matching objective, implemented with Hungarian matching where needed. The resulting reconstruction score is used to predict whether the set is valid.

## Citations

If this code is used, please cite the accompanying SetMTr manuscript:

Fengnan Mi and Jing Huang, "Predicting the Validity of Set Data with Self-supervised Masked Transformer."

Relevant external sources include:

- Rong, C., Liu, Z., Huo, N., and Sun, H. "Exploring Chinese Dietary Habits Using Recipes Extracted From Websites." IEEE Access, 2019. DOI: https://doi.org/10.1109/ACCESS.2019.2900504.
- Hahsler et al., `arules`: Mining Association Rules and Frequent Itemsets, R package: https://cran.r-project.org/package=arules.
- LeCun et al., MNIST database: http://yann.lecun.com/exdb/mnist/.
- Princeton ModelNet: https://modelnet.cs.princeton.edu/.
- BiGG Models: https://bigg.ucsd.edu/.
- Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks." ICML, 2019.
- Zaheer et al., "Deep Sets." NeurIPS, 2017.

## License & Contribution Guidelines

No public reuse license is included in the current code package. The code should therefore be treated as provided for manuscript review and reproducibility inspection unless the authors add an explicit license.

Before public release, the authors should add:

- A license file, such as MIT, BSD-3-Clause, or Apache-2.0.
- A `requirements.txt` file matching the tested environment.
- A reproduction script or command list for the simulated datasets, as stated in the manuscript's data and code availability section.
- Clear instructions for obtaining third-party datasets that cannot be redistributed.

Suggested contribution workflow for future public development:

- Open an issue to discuss bugs, dataset problems, or feature requests.
- Submit focused pull requests with clear descriptions of the changes.
- Include reproduction steps and configuration changes for experiment-related contributions.
- Avoid committing large raw datasets unless redistribution is permitted by the original data provider.
