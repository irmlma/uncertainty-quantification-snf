from collections import namedtuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.training.early_stopping import EarlyStopping
from jax import random
from rmsyutls import as_batch_iterators


def train(*, rng_key, data, model, config):
    prng_seq = hk.PRNGSequence(rng_key)

    train_iter, val_iter = as_batch_iterators(
        next(prng_seq),
        namedtuple("named_dataset", "y")(data),
        config.training.batch_size,
        config.training.train_val_split,
        config.training.shuffle_data,
    )
    params = model.init(next(prng_seq), **train_iter(0))

    adam = optax.adamw(
        learning_rate=config.optimizer.params.learning_rate,
        b1=config.optimizer.params.b1,
        b2=config.optimizer.params.b2,
        eps=config.optimizer.params.eps,
        weight_decay=config.optimizer.params.weight_decay,
    )
    opt_state = adam.init(params)

    @jax.jit
    def step(params, opt_state, **batch):
        def loss_fn(params):
            return -jnp.sum(model.apply(params, **batch))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = adam.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state

    losses = np.zeros([config.training.n_iter, 2])
    best_params = params
    best_loss = np.inf
    best_itr = 0
    logging.info("training model")

    early_stop = EarlyStopping(min_delta=10, patience=3)
    idxs = train_iter.idxs
    for i in range(config.training.n_iter):
        train_loss = 0.0
        idxs = random.permutation(next(prng_seq), idxs)
        for j in range(train_iter.num_batches):
            batch = train_iter(j, idxs)
            batch_loss, params, opt_state = step(params, opt_state, **batch)
            train_loss += batch_loss
        validation_loss = _validation_loss(params, model, val_iter)
        losses[i] = jnp.array([train_loss, validation_loss])
        logging.info(
            f"epoch {i} train/val loss : {train_loss}/{validation_loss}"
        )

        _, early_stop = early_stop.update(validation_loss)
        if validation_loss is jnp.nan:
            logging.warning("found nan validation loss. breaking")
            break
        if early_stop.should_stop:
            logging.info("Met early stopping criterion, breaking...")
            break
        if validation_loss < best_loss:
            logging.info("new best loss found at epoch: %d", i)
            best_params = params.copy()
            best_loss = validation_loss
            best_itr = i

    losses = jnp.vstack(losses)[:i, :]
    return {
        "params": best_params,
        "loss": best_loss,
        "itr": best_itr,
        "losses": losses,
        "config": config,
    }


def _validation_loss(params, model, val_iter):
    @jax.jit
    def _loss_fn(**batch):
        return -jnp.sum(model.apply(params, **batch))

    losses = jnp.array(
        [_loss_fn(**val_iter(j)) for j in range(val_iter.num_batches)]
    )
    return jnp.sum(losses)
