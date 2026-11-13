# Neural Network Quantization Study

This project studies how a conventional neural network's accuracy changes as its
weights and activations are represented with fewer bits. It starts with simple,
data-agnostic quantization and progressively uses calibration data and model
behavior to make better quantization choices.

## Starting experiment

- **Dataset:** MNIST, with fixed train/validation/test splits and normalization.
- **Model:** a small fully connected classifier (784 → 256 → 128 → 10), trained
  in FP32 as the reference model.
- **Precision ladder:** FP32 reference, FP16 and BF16, FP8 (E4M3 and E5M2),
  then integer weights at INT16, INT8, INT4, INT2, and 1-bit. Include INT32 as
  a wide integer reference / accumulator where relevant, not as a low-bit model
  compression point. Some formats require emulation if the chosen hardware or
  framework does not support them natively.
- **Primary result:** test accuracy and accuracy lost relative to the FP32 model.
  Also report model storage estimates and quantization settings so runs can be
  compared fairly.

## Experiment progression

1. **FP32 reference:** train once, record baseline metrics, and save the model.
2. **Floating-point reduced precision:** evaluate FP16, BF16, and supported FP8
   formats against FP32. Record the FP8 format and scaling policy because E4M3
   and E5M2 have different range/precision tradeoffs.
3. **Blind integer weight quantization:** apply the same symmetric uniform
   quantizer to every weight tensor, with no calibration examples and no
   layer-specific adjustment. Sweep INT16, INT8, INT4, INT2, and 1-bit where
   meaningful. This establishes the simplest quantization baseline.
4. **Layer-wise weight quantization:** keep the same quantizer but choose scales
   per layer rather than globally; compare per-tensor and per-channel scales.
5. **Activation-calibrated quantization:** collect activation statistics from a
   calibration split, then quantize activations using observed ranges. Compare
   min/max ranges with robust clipping (for example, percentile clipping).
6. **Activation-informed weight choices:** use calibration activations to
   estimate which layers are most sensitive to weight quantization. Allocate
   more bits to sensitive layers and fewer to tolerant layers under a stated
   average-bit budget. This makes explicit how activation magnitudes and
   downstream error affect the choice of which weights need precision.
7. **Optional fine-tuning:** fine-tune the quantized model (QAT or a clearly
   specified quantization-aware fine-tuning variant) and report separately from
   post-training quantization results.

For every stage, evaluate the same held-out test set at each bit width. Keep
training data, calibration data, and test data separate; calibration must not
use test examples. Record seeds and quantization parameters alongside results.

## Comparisons to make

Plot test accuracy against effective bits per weight, with one curve per method.
Include the FP32 reference, and report per-layer bit allocations for mixed
precision. Clearly distinguish weight-only quantization from quantizing both
weights and activations. Storage estimates should include scale/zero-point
overhead where practical; latency is hardware- and kernel-dependent, so only
compare it when measured on the same runtime and device.

## Scope

The first milestone is a reproducible MNIST MLP experiment and the blind
weight-only baseline. More advanced calibration, mixed-precision allocation,
and quantization-aware fine-tuning build on that baseline rather than being
bundled into one opaque result.

## Repository layout

```text
configs/                    Shared/default run configuration
data/
  raw/                      Downloaded source data (not committed)
  processed/                Prepared splits and cached tensors (not committed)
docs/                       Experiment notes and design decisions
experiments/
  configs/                  Per-experiment overrides
  scripts/                  Reproducible run entry points
results/
  figures/                  Plots
  metrics/                  Machine-readable result summaries
src/
  data/                     Dataset loading and preprocessing
  models/                   FP32 reference model definitions
  training/                 Training and optional fine-tuning loops
  quantization/
    floating_point/         FP16, BF16, and FP8 paths
    integer/                 INT16 through 1-bit quantizers
    mixed_precision/         Sensitivity analysis and bit allocation
  evaluation/               Metrics and comparison utilities
```

Data, generated results, and experiment-specific code stay separate from the
reusable implementation in `src/`.
