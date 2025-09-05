# Installation

We recommend installing in a clean conda environment. Below are quick recipes for the three most common setups.

## Step-by-step

### 1. Create environment
```bash
conda config --add channels conda-forge
conda create -n jax python=3.12
conda activate jax
```

### 2. Install CUDA (Linux + NVIDIA GPU only)
```bash
conda install cuda -c nvidia
```

### 3. Install JAX (platform-specific)
Follow the official [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Use the variant matching your system:
```bash
# Linux / Windows (CPU)
pip install "jax[cpu]"

# Linux + CUDA 12
pip install "jax[cuda12]"

# macOS (Apple Silicon, Metal)
pip install "jax[cpu]" "jax-metal"
```

### 4. Install radjax (developer/editable)
```bash
git clone https://github.com/tcig-cips/radjax.git
cd radjax
pip install -e .
```

#### Editable install notes (-e flag)
This project uses a modern `pyproject.toml` build (PEP 660). If you see:
```
A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.
```
upgrade your tooling:
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Optional: visualization + MCMC tools
```bash
conda install emcee h5py ffmpeg
```
