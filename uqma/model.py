import distrax
import haiku as hk
import jax
from absl import logging
from jax import numpy as jnp
from surjectors import (
    AffineMaskedCouplingInferenceFunnel,
    Chain,
    RationalQuadraticSplineMaskedCouplingInferenceFunnel,
    TransformedDistribution,
)
from surjectors.util import make_alternating_binary_mask


def _affine_mlp_conditioner(config, n_params, event_shape):
    activation = getattr(jax.nn, config.activation)
    net = hk.Sequential(
        [
            hk.nets.MLP(
                [config.ndim_hidden_layers] * config.n_hidden_layers,
                activate_final=True,
                activation=activation,
            ),
            hk.Linear(
                event_shape * n_params,
                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),
        ]
    )
    return net


def _conditioner_fn(
    config, transformer_type, num_transformer_params, event_shape
):
    activation = getattr(jax.nn, config.activation)
    if transformer_type == "nsf":
        n_params = 3 * num_transformer_params + 1
        net = hk.Sequential(
            [
                hk.nets.MLP(
                    [config.ndim_hidden_layers] * config.n_hidden_layers,
                    activate_final=True,
                    activation=activation,
                ),
                hk.Linear(
                    event_shape * n_params,
                    w_init=jnp.zeros,
                    b_init=jnp.zeros,
                ),
                hk.Reshape((event_shape,) + (n_params,), preserve_dims=-1),
            ]
        )
    else:
        net = _affine_mlp_conditioner(
            config, num_transformer_params, event_shape
        )
    return net


def _decoder_fn(config, event_shape):
    if config.type == "mlp":
        decoder_net = _affine_mlp_conditioner(
            config, config.n_params, event_shape
        )
    else:
        logging.fatal("didnt find correct decoder type")
        raise ValueError("didnt find correct decoder type")

    def _fn(z):
        params = decoder_net(z)
        mu, log_scale = jnp.split(params, 2, -1)
        return distrax.Independent(distrax.Normal(mu, jnp.exp(log_scale)))

    return _fn


def _bijector_fn(config):
    if config.type == "nsf":

        def _fn(params):
            return distrax.RationalQuadraticSpline(
                params, range_min=config.range_min, range_max=config.range_max
            )

    elif config.type == "affine":

        def _fn(params):
            means, log_scales = jnp.split(params, 2, -1)
            return distrax.ScalarAffine(means, jnp.exp(log_scales))

    else:
        raise ValueError("didnt find bijector")
    return _fn


def make_model(config, event_shape):
    def _flow(**kwargs):
        n_dimension = event_shape
        layers = []

        for i, entry in enumerate(config.n_flow_layers):
            if entry.type == "bijection":
                mask = make_alternating_binary_mask(n_dimension, i % 2 == 0)
                layer = distrax.MaskedCoupling(
                    mask=mask,
                    bijector=_bijector_fn(entry.transformer),
                    conditioner=_conditioner_fn(
                        entry.conditioner,
                        entry.transformer.type,
                        entry.transformer.n_params,
                        n_dimension,
                    ),
                )
            elif entry.type == "funnel" and entry.transformer.type == "affine":
                latent_dim = int(entry.reduction_factor * n_dimension)
                layer = AffineMaskedCouplingInferenceFunnel(
                    latent_dim,
                    decoder=_decoder_fn(
                        entry.decoder, n_dimension - latent_dim
                    ),
                    conditioner=_conditioner_fn(
                        entry.conditioner,
                        entry.transformer.type,
                        entry.transformer.n_params,
                        n_dimension,
                    ),
                )
                n_dimension = latent_dim
            elif entry.type == "funnel" and entry.transformer.type == "nsf":
                latent_dim = int(entry.reduction_factor * n_dimension)
                layer = RationalQuadraticSplineMaskedCouplingInferenceFunnel(
                    latent_dim,
                    decoder=_decoder_fn(
                        entry.decoder, n_dimension - latent_dim
                    ),
                    conditioner=_conditioner_fn(
                        entry.conditioner,
                        entry.transformer.type,
                        entry.transformer.n_params,
                        n_dimension,
                    ),
                    range_min=-5.0,
                    range_max=5.0,
                )
                n_dimension = latent_dim
            else:
                logging.fatal("didnt find correct flow type")
                raise ValueError("didnt find correct flow type")
            layers.append(layer)

        chain = Chain(layers)
        base_distribution = distrax.Independent(
            distrax.Normal(
                loc=jnp.zeros(n_dimension), scale=jnp.ones(n_dimension)
            ),
            reinterpreted_batch_ndims=1,
        )
        td = TransformedDistribution(base_distribution, chain)
        return td.log_prob(**kwargs)

    td = hk.transform(_flow)
    td = hk.without_apply_rng(td)
    return td
