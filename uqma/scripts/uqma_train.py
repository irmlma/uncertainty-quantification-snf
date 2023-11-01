import pickle

import jax
import pandas as pd
from absl import app, flags
from ml_collections import config_flags

from uqma import make_model, train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "training configuration", lock_config=False
)
flags.DEFINE_string(
    "infile",
    None,
    "csv-separated input file containing pen-ultimate layer featurs",
)
flags.DEFINE_string("outfile", None, "name of the output file")

flags.mark_flags_as_required(["config", "infile", "outfile"])


def _train(argv):
    del argv
    config = FLAGS.config

    model = make_model(config)
    features = pd.read_csv(FLAGS.infile).values
    obj = train(
        rng_key=config.training.rng_seq_key,
        data=features,
        model=model,
        config=config,
    )

    with open(FLAGS.outfile, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    jax.config.config_with_absl()
    app.run(_train)
