# Uncertainty quantification for mobility analysis

[![ci](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml/badge.svg)](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2311.00377-b31b1b.svg)](https://arxiv.org/abs/2311.00377)

## About

This repository contains library code for training a surjective normalizing flow for out-of-distribution detection using epistemic uncertainty estimates.

## Installation

To install the latest GitHub <TAG>, just call the following on the
command line:

```bash
pip install git+https://github.com/irmlma/uncertainty-quantification-for-mobility-analysis@<TAG>
```

This installs the library as well as executables in your current (virtual) environment.

## Example usage

Having installed as described above you can train and make predictions using the provided executables.

```bash
docker build . -t uqma
```

You can then run the container using

```bash
docker run uqma --help
```

Train the model using the provided config file via:
```bash
docker run uqma \
  --mode=train \
  --config=<<config.py>> \
  --infile=<<train_dataset.csv>> \
  --outfile=<<outfile.pkl>>
```

where
- `<<config.py>>` is a config file that is following the template in `configs/config.py`,
- `<<train_dataset.csv>>` is a comma-separated file of numerical values which correspond to the features obtained from transforming inputs through a neural network,
- `<<outfile.pkl>>` is the outfile to which parameter and meta data is saved.

To make predictions for epistemic uncertainty estimates, call:
```bash
docker run uqma \
  --mode=predict \
  --config=<<config.py>> \
  --infile=<<test_dataset.csv>> \
  --outfile=<<outfile.pkl>> \
  --checkpopint=<<checkpoint>>
```

where
- `<<config.py>>` is the same as above,
- `<<test_dataset.csv>>` is a data set for which you want to evaluate if it is OoD,
- `<<outfile.pkl>>` is the name of the outfile,
- `<<checkpoint>>` is the parameter file obtained through the training (i.e., in this case `<<outfolder>>/params.pkl`).

## Citation

If you find our work relevant to your research, please consider citing

```
@article{dirmeier2023uncertain,
  title={Uncertainty quantification and out-of-distribution detection using surjective normalizing flows},
  author={Simon Dirmeier and Ye Hong and Yanan Xin and Fernando Perez-Cruz},
  year={2023},
  journal={arXiv preprint arXiv:2311.00377}
}
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
