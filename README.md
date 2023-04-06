# Sequel: A Continual Learning Library in PyTorch and JAX

The goal of this library is to provide a simple and easy to use framework for continual learning. The library is written in PyTorch and JAX and provides a simple interface to run experiments. The library is still in development and we are working on adding more algorithms and datasets.

- Documetation: https://nik-dim.github.io/sequel-site/
- Reproducibility Board: https://nik-dim.github.io/sequel-site/reproducibility/ 
- Weights&Biases: https://wandb.ai/nikdim/SequeL 
## Installation

The library can be installed via pip:
```bash
pip install sequel-core
```

Alternatively, you can install the library from source:
```bash
git clone https://github.com/nik-dim/sequel.git
python3 -m build
```

or use the library by cloning the repository. In order to use the library, you need to install the dependencies. This can be done via the `requirements.txt` file. We recommend to use a conda environment for this. The following commands will create a conda environment with the required packages and activate it:
```bash
# create the conda environment
conda create -n sequel -y python=3.10 cuda cudatoolkit cuda-nvcc -c nvidia -c anaconda
conda activate sequel 

# install all required packages
pip install -r requirements.txt

# Optional: Depending on the machine, the next command might be needed to enable CUDA support for GPUs
pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Run an experiment

For some examples, you can modify the `example_pytorch.py` and `example_jax.py` files, or run:
```bash
# example experiment on PyTorch
python example_pytorch.py

# ...and in JAX
python example_jax.py
```

Experiments are located in the `examples/` directory in `configs`. In order to run an experiment you simply do the following:

```bash
python main.py +experiment=EXPERIMENT_DIR/EXPERRIMENT

# examples
python main.py +examples=ewc_rotatedmnist       mode=pytorch        # or mode=jax
python main.py +examples=mcsgd_rotatedmnist     mode=pytorch        # or mode=jax
```

In order to create your own experiment you follow the template of the experiments in `configs/examples/`. You override the defaults so that e.g. another algorithm is selected and you specify the training details. To run multiple experiments with different configs, the `--multirun` flag of [Hydra](https://hydra.cc/docs) can be used. 
For instance:
```bash
python main.py --multirun +examples=ewc_rotatedmnist \
     mode=pytorch optimizer.lr=0.01,0.001 \
     benchmark.batch_size=128,256 \ 
     training.epochs_per_task=1 # online setting
```
