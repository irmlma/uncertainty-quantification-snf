import pickle

import jax
import pandas as pd
from absl import app, flags
from ml_collections import config_flags

from uqma import make_model, predict

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "params", None, "pickle file with training parameters", lock_config=False
)
flags.DEFINE_string(
    "infile",
    None,
    "csv-separated input file containing pen-ultimate layer features",
)
flags.DEFINE_string("outfile", None, "name of the output file")

flags.mark_flags_as_required(["params", "infile", "outfile"])


def _predict(argv):
    del argv

    with open(FLAGS.params, "rb") as fh:
        obj = pickle.load(fh)
    config = obj["config"]

    model = make_model(config)
    features = pd.read_csv(FLAGS.infile).values
    lps = predict(
        rng_key=config.prediction.rng_seq_key,
        params=obj["params"],
        model=model,
        data=features,
        config=config,
    )

    with open(FLAGS.outfile, "wb") as fh:
        pickle.dump(lps, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main(*args, **kwargs):
    jax.config.config_with_absl()
    app.run(_predict)
