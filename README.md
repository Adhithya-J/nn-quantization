# Pretrained Neural Network Quantization Study

This project starts from a standard pretrained neural network and studies
post-training weight quantization. It does not train or fine-tune the model.

## Current experiment

- **Model:** torchvision ResNet-18 with its published ImageNet-1K pretrained
  weights (`ResNet18_Weights.DEFAULT`). The weights are downloaded from the
  official torchvision checkpoint host on first run and cached by torchvision.
- **Quantization:** blind symmetric global-scale weight quantization at INT16,
  INT8, INT4, INT2, and 1-bit. Biases, batch normalization, and activations stay FP32.
  One-bit uses mean-absolute scaling (XNOR-Net style) to prevent activation overflow.
  Quantized weights are dequantized to FP32 for inference, so this measures the effect of
  quantization; it does not run packed integer kernels or measure latency.
- **Selective format:** an activation-informed mixed precision format. It
  forwards separate Imagenette training images through the network, measures
  how often each layer input channel is active, then assigns 6 bits to more
  active input-channel weight groups and 4 bits to less active groups. This
  gives about 4.5 bits per quantized weight on average, similar in spirit to a
  mixed-block 4-bit format. It is a small research implementation, not a
  bit-compatible GGUF Q4_K_M/Q4_S_M encoder.
- **Test set:** a deterministic, balanced 500-image sample from Imagenette's
  labeled validation split (50 images per class). Accuracy is scored only among
  those 10 Imagenette classes, even though the pretrained network has 1,000
  output classes. This is the quick iteration setting; it does not run full
  ImageNet validation.

## Run

Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the default quantization sweep from the repository root:

```bash
python experiments/scripts/run_quantization.py
```

On first use, torchvision downloads the pretrained ResNet-18 checkpoint, and
the run downloads Imagenette into `data/raw/`. A copy of the original model
state is also saved at `results/checkpoints/resnet18_pretrained_fp32.pt`, so
future runs can load it without contacting the model host. Imagenette is
published by fast.ai; its `train` split is used only for activation calibration
and its `val` split only for measurement. By default, a seeded sample selects
50 validation images per class; edit `test_images_per_class` to adjust the
quick test size. The loader verifies the archive checksum and maps folders to
the matching ImageNet-1K outputs. Images use the preprocessing transforms
published with the pretrained weights.

Results are written to
`results/metrics/pretrained_resnet18_quantization.json`. The report measures
the original model and every quantized copy on the same 500 Imagenette images,
and includes 10-class test accuracy, accuracy drop, quantization settings,
parameter count, calibration/test sample counts, and estimated storage.
Storage is an estimate and excludes container/alignment overhead. In addition to blind
global scaling, the repository implements activation-selected mixed precision across
multiple bit pairs, per-channel symmetric quantization with OMSE clipping, block-level
group-wise quantization (symmetric and asymmetric), and HAWQ/Fisher-style sensitivity-guided
mixed-precision budget allocation that pushes sub-3-bit and 2-bit frontiers. All methods keep
batch-normalization parameters and biases in FP32.

### Latest quick comparison

Measured on 500 images (50 per class), with predictions restricted to the 10
Imagenette classes:

| Family | Method | Effective Bits | Top-1 Accuracy | Accuracy Drop | Loss | Storage |
|---|---|---:|---:|---:|---:|---:|
| **Baseline** | Original FP32 | 32.00 | 97.4% | — | 0.0726 | 46.76 MB |
| **Sensitivity-Guided Pareto** | Pareto Budget 3.5b | 3.53 | **98.2%** | -0.8 points | 0.0679 | 7.02 MB |
| **Sensitivity-Guided Pareto** | Pareto Budget 3.0b | 3.03 | **98.0%** | -0.6 points | 0.0622 | 6.29 MB |
| **Sensitivity-Guided Pareto** | Pareto Budget 2.5b | 2.53 | **97.8%** | -0.4 points | 0.0711 | 5.56 MB |
| **Sensitivity-Guided Pareto** | Pareto Budget 2.2b | 2.25 | **95.6%** | 1.8 points | 0.1181 | 5.14 MB |
| **Sensitivity-Guided Pareto** | Pareto Budget 2.0b | 2.04 | **92.8%** | 4.6 points | 0.2137 | 4.85 MB |
| **Group-wise (AWQ / GPTQ / GGUF)** | Group-wise INT4 ($G=32$, Asym) | 4.00 | 96.6% | 0.8 points | 0.1125 | 7.71 MB |
| **Group-wise (AWQ / GPTQ / GGUF)** | Group-wise INT4 ($G=32$, Sym) | 4.00 | 95.4% | 2.0 points | 0.1370 | 7.34 MB |
| **Group-wise (AWQ / GPTQ / GGUF)** | Group-wise INT4 ($G=64$, Sym) | 4.00 | 95.2% | 2.2 points | 0.1299 | 6.61 MB |
| **Group-wise (AWQ / GPTQ / GGUF)** | Group-wise INT3 ($G=32$, Asym) | 3.00 | 91.8% | 5.6 points | 0.2316 | 6.25 MB |
| **Group-wise (AWQ / GPTQ / GGUF)** | Group-wise INT3 ($G=32$, Sym) | 3.00 | 82.6% | 14.8 points | 0.5472 | 5.88 MB |
| **Per-Channel + OMSE** | Per-Channel INT4 (MSE Clipped) | 4.00 | 93.0% | 4.4 points | 0.2477 | 5.91 MB |
| **Per-Channel (PyTorch / TensorRT)** | Per-Channel INT4 | 4.00 | 89.6% | 7.8 points | 0.3257 | 5.91 MB |
| **Per-Channel (PyTorch / TensorRT)** | Per-Channel INT8 | 8.00 | 97.6% | -0.2 points | 0.0738 | 11.74 MB |
| **Activation-Selected Mixed** | Mixed 4/8-bit | 5.00 | 97.6% | -0.2 points | 0.0743 | 7.36 MB |
| **Activation-Selected Mixed** | Mixed 4/6-bit (GGUF Q4_K_M) | 4.50 | 96.4% | 1.0 point | 0.0951 | 6.63 MB |
| **Activation-Selected Mixed** | Mixed 2/4-bit | 2.50 | 13.6% | 83.8 points | 6.8182 | 3.71 MB |
| **Blind Global** | Global INT8 | 8.00 | 97.8% | -0.4 points | 0.0778 | 11.72 MB |
| **Blind Global** | Global INT4 | 4.00 | 10.0% | 87.4 points | 69.05 | 5.88 MB |
| **Blind Global** | Global INT2 | 2.00 | 10.0% | 87.4 points | 2.3027 | 2.96 MB |
| **Blind Global** | Global INT1 (XNOR-Net scale) | 1.00 | 10.0% | 87.4 points | 2.3131 | 1.50 MB |

## Future scope

The only supported model is pretrained torchvision ResNet-18 and the current
test set is Imagenette. Model training and fine-tuning are out of scope. Other
pretrained architectures and test datasets (including Fashion-MNIST and full
ImageNet), improved activation sensitivity scores, group-size and bit-allocation
experiments, activation quantization, and packed integer inference are future
extensions.

## Repository layout

```text
experiments/configs/         Run configuration
experiments/scripts/         Quantization entry point
src/data/                    Optional validation data adapter
src/evaluation/              Evaluation metrics
src/models/                  Pretrained model loader
src/quantization/integer/    Integer weight quantizer
results/metrics/             Generated machine-readable results
```
