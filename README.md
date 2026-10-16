# Oscillatory flow matching

This project is being implemented to test an attention-free Kuramoto oscillator velocity network for unconditional image flow matching. The first study uses MNIST for development and CIFAR-10 for a fixed-budget comparison with a small transformer and recurrent convolutional control. No performance advantage is assumed.

The pre-specified methodology and budget are in [docs/protocol.md](docs/protocol.md). Installation and working commands will be verified and documented before the first training run.

## Environment

Python 3.12, uv, PyTorch, torchvision, TorchCFM, torchdiffeq, torch-fidelity, SciPy, safetensors, Hugging Face Hub, Pandoc, and Typst. The local CUDA configuration targets an RTX 5070 laptop GPU with 8 GB VRAM.

```bash
uv sync --extra cuda
# CPU-only environments: uv sync --extra cpu
cp .env.example .env
```

This repository contains original implementation code. Method references and provenance are recorded in the protocol and module comments.
