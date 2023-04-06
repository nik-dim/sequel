---
hide:
  - toc
---

# Installation


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