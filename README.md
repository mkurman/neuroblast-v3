# NeuroBLAST V3
A novel hybrid model designed with a biologically inspired "cortical" structure.

## Architecture Highlights
NeuroBLAST differs from standard Transformers by utilizing a three-stage cortical design:

- **Sensory Cortex**: A hybrid stage designed for initial feature extraction. It alternates between standard **Self-Attention** layers (for global context) and **Dilated Causal 2D Convolutions** (for local patterns). The convolutions use exponentially increasing dilation factors to capture multi-scale features effectively.
- **Associative Cortex**: The core reasoning engine. It continues the hybrid Attention/Convolution pattern but introduces **Alternating RoPE**: standard attention layers alternate with "No-RoPE" attention layers, encouraging the model to balance between position-dependent and purely semantic (content-based) processing.
- **Motor Cortex**: The output generation stage. It consists entirely of **Self-Attention** layers to refine the final representations. Like the Associative Cortex, it employs the **Alternating RoPE** strategy.
- **Deep Residual Bridges**: Specialized long-range connections that mitigate signal degradation deep in the network.
    - Between Sensory and Associative stages: Injects the **original embeddings** (normalized and activated types).
    - Between Associative and Motor stages: Injects the **negated original embeddings**, a mechanism designed to provide contrastive grounding or subtractive reference to the initial context.


![NeuroBLAST Architecture](assets/architecture.jpeg)

## Implementation Details

PyTorch and JAX implementations are available in the `model` directory.

Early checkpoint: [NeuroBLAST V3 SYNTH EC 150000](https://huggingface.co/mkurman/NeuroBLAST-V3-SYNTH-EC-150000) trained on the [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) dataset.

### Model Configuration
The default configuration used in training:
- **Hidden Size**: 512
- **Intermediate Size**: 3072
- **Total Layers**: 72
  - Associative: 32
  - Sensory: 24
  - Motor: 16
- **Attention Heads**: 16
- **Key/Value Heads**: 8
- **Head Dimension**: 128
- **Max Position Embeddings**: 32768

## Installation

### Prerequisites
- Python >= 3.10

### Standard Installation (PyTorch)
```bash
uv pip install -e .
```

### JAX / TPU Installation
For JAX training on TPU, we recommend installing dependencies in the following order to ensure compatibility:

```bash
# Install JAX with TPU support
uv pip install -U jax[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html

# Install PyTorch with XLA support
uv pip install "torch==2.8.0" "torch_xla[tpu]==2.8.1" -f https://storage.googleapis.com/libtpu-releases/index.html

# Install package with JAX extras
uv pip install -e ".[jax]"
```

## Dataset & Tokenizer

- **Dataset**: [PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) (Common Corpus Synthetic data)
- **Tokenizer**: [PleIAs/Baguettotron](https://huggingface.co/PleIAs/Baguettotron)

## Training

### PyTorch Training
To launch the PyTorch training script:

```bash
python train/train.py
```

This script uses `transformers.Trainer` and `accelerate` for distributed training support. It defaults to training on GPU.

### JAX Training
To launch the JAX training script (optimized for TPU):

```bash
python train/train_jax.py
```

Arguments for distributed training (like `JAX_COORDINATOR_ADDRESS`, `JAX_PROCESS_COUNT`, `JAX_PROCESS_INDEX`) can be set via environment variables. The script supports gradient accumulation and uses `orbax-checkpoint` for saving models.
