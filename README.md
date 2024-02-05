# Uncertainty quantification for mobility analysis

[![ci](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml/badge.svg)](https://github.com/irmlma/uncertainty-quantification-snf/actions/workflows/ci.yaml)
[![arXiv](https://img.shields.io/badge/arXiv-2311.00377-b31b1b.svg)](https://arxiv.org/abs/2311.00377)

## About

This repository contains library code for training a surjective normalizing flow for out-of-distribution detection using epistemic uncertainty estimates.

## Installation

To install the latest GitHub <TAG>, just call the following on the command line:

```bash
docker build https://github.com/irmlma/uncertainty-quantification-snf.git#<TAG> -t uqma
```

where <TAG> is, e.g., `v0.1.1`.

## Example usage

Having installed as described above you can train and make predictions using the provided executables.

You can then run the container using

```bash
docker run uqma --help
```

Train the model using the provided config file via:
```bash
docker run -v <<some path>>:/mnt \
  uqma \
  --mode=train \
  --config=/mnt/<<config.py>> \
  --infile=/mnt/<<train_dataset.csv>> \
  --outfile=/mnt/<<outfile.pkl>>
```

where
- `<<some path>` is a local path you want to mount to `/mnt/` to make it accessible to Docker,
- `<<config.py>>` is a config file that is following the template in `configs/config.py`,
- `<<train_dataset.csv>>` is a comma-separated file of numerical values which correspond to the features obtained from transforming inputs through a neural network,
- `<<outfile.pkl>>` is the outfile to which parameter and meta data is saved.

To make predictions for epistemic uncertainty estimates, call:
```bash
docker run -v <<some path>>:/mnt \
  uqma \
  --mode=predict \
  --config=/mnt/<<config.py>> \
  --infile=/mnt/<<test_dataset.csv>> \
  --outfile=/mnt/<<outfile.pkl>> \
  --checkpopint=/mnt/<<checkpoint>>
```

where
- `<<some path>` is the same as above,
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
