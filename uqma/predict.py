from collections import namedtuple

from absl import logging
from jax import jit
from jax import numpy as jnp
from rmsyutls import as_batch_iterator


def predict(rng_key, params, model, data, batch_size):
    @jit
    def _predict(**batch):
        return model.apply(params, **batch)

    logging.info("making predictions")
    itr = as_batch_iterator(
        rng_key,
        namedtuple("named_dataset", "y")(data),
        batch_size,
        False,
    )
    lps = [None] * itr.num_batches
    for i in range(itr.num_batches):
        lps[i] = _predict(**itr(i))
    lps = jnp.concatenate(lps)[: len(data)]
    return lps
