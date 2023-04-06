import sys
from pathlib import Path

# setting path
sys.path.append(Path().parent.absolute().as_posix())
print(sys.path)

import jax
import jax.numpy as jnp
from sequel.backbones.jax import MLP, CNN


def test_mlp():
    init_rng = jax.random.PRNGKey(0)
    x = jnp.ones((10, 1, 28, 28))

    model = MLP(width=2, n_hidden_layers=3)
    params = model.init(init_rng, x=x)
    y = model.apply(params, x)
    assert y.shape == (10, 10)

    model = MLP(width=2, n_hidden_layers=3, num_classes=123)
    params = model.init(init_rng, x=x)
    y = model.apply(params, x)
    assert y.shape == (10, 123)


# def test_cnn():
#     init_rng = jax.random.PRNGKey(0)
#     x = jnp.ones((10, 1, 28, 28))

#     model = CNN(channels=[10, 10])
#     params = model.init(init_rng, x=x)
#     y = model.apply(params, x)
#     assert y.shape == (10, 10)

#     model = CNN(channels=[10, 10], num_classes=123)
#     params = model.init(init_rng, x=x)
#     y = model.apply(params, x)
#     assert y.shape == (10, 123)

#     model = CNN(channels=[10, 10], num_classes=123, linear_layers=20)
#     params = model.init(init_rng, x=x)
#     y = model.apply(params, x)
#     assert y.shape == (10, 123)
