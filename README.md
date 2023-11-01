# Uncertainty quantification for mobility analysis

[![ci](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml/badge.svg)](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml)

## About

This repository contains library code for training a surjective normalizing flow for out-of-distribution detection using epistemic uncertainty estimates.

## Installation

To install the latest GitHub <TAG>, just call the following on the
command line:

```bash
pip install git+https://github.com/irmlma/uncertainty-quantification-for-mobility-analysis@<TAG>
```

This installs the library as well as executables in your current (virtual) environment.

## Usage

Having installed as described above you can train and make predictions using the provided exectuables.

Train the model using the provided config file via:

```{python}
uqma-train
    --config=configs/config.py \
    --infile=<FILE> \
    --outfile=<PARAMS_FILE>
```

where <FILE> is a COMMA-separated file of numerical values which correspond to the features obtained from transforming inputs through a
neural network that has residual connections and was trained using spectral-normalization and <PARAMS_FILE> is some file to which parameters are written (see Dirmeier et al. (2023)).

To make predictions for epistemic uncertainty estimates, call:

```{python}
uqma-train
    --params=<PARAMS_FILE> \
    --infile=<FILE> \
    --outfile=<PARAMS_FILE>
```

where <PARAMS_FILE> is the same file as before, <FILE> is a features file and <OUTFILE> is a file where results are written to.

## Citation

If you find our work relevant to your research, please consider citing using the following reference:

```
@article{dirmeier2023uncertain,
  title={Uncertainty quantification and out-of-distribution detection using surjective normalizing flows},
  author={Simon Dirmeier and Ye Hong and Yanan Xin and Fernando Perez-Cruz},
  year={2023},
  journal={arXiv preprint}
}
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
