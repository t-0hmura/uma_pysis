# **UMA - Pysisyphus Interface**

## 1. Installation Guide

For CUDA 12.6:
```bash
pip install git+https://github.com/t-0hmura/uma_pysis.git
huggingface-cli login
```

For CUDA 12.8 (required for RTX 50 series):
```bash
pip install git+https://github.com/t-0hmura/uma_pysis.git
pip install --force-reinstall torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
huggingface-cli login
```

