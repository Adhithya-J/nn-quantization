# Pretrained Neural Network Quantization Study

This project starts from a standard pretrained neural network and studies
post-training weight quantization. It does not train or fine-tune the model.

## Current experiment

- **Model:** torchvision ResNet-18 with its published ImageNet-1K pretrained
  weights (`ResNet18_Weights.DEFAULT`). The weights are downloaded from the
  official torchvision checkpoint host on first run and cached by torchvision.
- **Quantization:** blind symmetric global-scale weight quantization at INT16,
  INT8, INT4, INT2, and 1-bit. Biases and activations stay FP32. Quantized
  weights are dequantized to FP32 for inference, so this measures the effect of
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
Storage is an estimate and excludes container/alignment overhead. The blind
baseline uses one global scale; the selective method uses one scale per
input-channel group and leaves batch-normalization weights in FP32.

### Latest quick comparison

Measured on 500 images (50 per class), with predictions restricted to the 10
Imagenette classes:

| Method | Effective weight bits | Top-1 accuracy | Accuracy drop |
| --- | ---: | ---: | ---: |
| Original FP32 | 32 | 97.4% | — |
| Blind global INT8 | 8 | 97.0% | 0.4 points |
| Blind global INT4 | 4 | 10.4% | 87.0 points |
| Activation-selected 4/6-bit groups | 4.50 | 96.4% | 1.0 point |
| Blind global INT2 | 2 | 10.0% | 87.4 points |
| Blind global INT1 | 1 | Non-finite outputs | — |

The mixed method keeps more-active input-channel groups at 6 bits and uses 4
bits for the rest. In this quick run it retained 96.4% accuracy with an
estimated 6.63 MB parameter storage, compared with 46.76 MB for FP32. These
numbers are an initial 500-image measurement, not a claim about full ImageNet
validation accuracy.

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
