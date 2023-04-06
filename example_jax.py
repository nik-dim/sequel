from sequel import benchmarks, backbones, algos, loggers, callbacks
import optax as tx

if __name__ == "__main__":
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
        loggers=[loggers.WandbLogger(disabled=True)],
        # algorithm-specific arguments
        ewc_lambda=1,
    )

    # start training
    algo.fit(epochs=1)
