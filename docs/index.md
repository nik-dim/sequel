# SequeL

## Getting started

### Installation

=== "from pip"
    ```bash
    pip install sequel-core
    ```

=== "from source"
    ```bash
    git clone https://github.com/nik-dim/sequel.git
    python3 -m build
    ```

The library can also be used by cloning the repository.

## Project layout

```txt
mkdocs.yml      # The configuration file for the documentation.
docs/
    index.md    # The documentation homepage.
    ...         # Other markdown pages, images and other files.
    
sequel/         # The source code lies here.
    algos/      # The Continual Learning Algorithms, e.g. EWC.
    backbones/  # The Neural Net classes.
    benchmarks/ # The benchmarks such as SplitMNIST.
    utils/      # Utility functions such as logging, callbacks etc.
```   


## Examples
The API for both JAX and PyTorch is the same. In the following example, we only need to change `pytorch` to `jax` 
and define the optimizer in a framework-specific way.

=== "PyTorch"

    ```python
    from sequel import benchmarks, backbones, algos, loggers, callbacks
    import torch

    # define the Continual Learning benchmark.
    benchmark = benchmarks.PermutedMNIST(num_tasks=3, batch_size=512)

    # define the backbone model, i.e., the neural network, and the optimizer
    backbone = backbones.pytorch.MLP(width=256, n_hidden_layers=2, num_classes=10)
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)

    # initialize the algorithm
    algo = algos.pytorch.EWC(
        backbone=backbone,
        optimizer=optimizer,
        benchmark=benchmark,
        callbacks=[
            callbacks.PyTorchMetricCallback(),
            callbacks.TqdmCallback(),
        ],
        loggers=[loggers.WandbLogger()],
        # algorithm-specific arguments
        ewc_lambda=1,
    )

    # start training
    algo.fit(epochs_per_task=1)

    ```

=== "JAX"

    ``` python
    from sequel import benchmarks, backbones, algos, loggers, callbacks
    import optax as tx

    # define the Continual Learning benchmark.
    benchmark = benchmarks.PermutedMNIST(num_tasks=3, batch_size=512)

    # define the backbone model, i.e., the neural network, and the optimizer
    backbone = backbones.jax.MLP(width=256, n_hidden_layers=2, num_classes=10)
    optimizer = tx.inject_hyperparams(tx.sgd)(learning_rate=0.1)

    # initialize the algorithm
    algo = algos.jax.EWC(
        backbone=backbone,
        optimizer=optimizer,
        benchmark=benchmark,
        callbacks=[
            callbacks.JaxMetricCallback(),
            callbacks.TqdmCallback(),
        ],
        loggers=[loggers.WandbLogger()],
        # algorithm-specific arguments
        ewc_lambda=1,
    )

    # start training
    algo.fit(epochs_per_task=1)

    ```