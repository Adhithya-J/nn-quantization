# Neural Network Quantization: A Socratic Study

This repository presents an inquiry-driven, **Socratic journey through post-training weight quantization (PTQ)** on a pretrained vision model (torchvision ResNet-18 on ImageNet/Imagenette). 

Rather than treating quantization as an opaque black box, the project progresses step-by-step from naive discretization to cutting-edge Pareto-optimal sub-3-bit compression—uncovering failure modes, mathematical root causes, and architectural trade-offs along the way.

---

## Core Fundamentals: Signal Dynamics & Quantization Mechanics

Before diving into empirical comparisons, it is essential to understand the mathematical physics of deep neural networks:

### 1. Signal Propagation & Layer Gain
In an $L$-layer network, activations evolve as $x_{l+1} = \sigma(W_l x_l + b_l)$. Under standard initialization (He / Xavier), the weights are distributed such that the expected per-layer variance gain $g_l \approx 1$:
$$\|x_L\| \approx \left(\prod_{l=1}^{L} g_l\right) \|x_0\| = g^L \|x_0\|$$

### 2. The Quantization Grid & The Zero-Bin
Quantizing weights to $b$ bits maps continuous values into $2^b$ discrete levels spaced by step size $s$:
$$s = \frac{\beta - \alpha}{2^b - 1}, \quad q = \text{clamp}\left(\left\lfloor \frac{w}{s} \right\rceil + z, 0, 2^b - 1\right), \quad \hat{w} = (q - z) \cdot s$$
Any weight whose magnitude falls within the zero-bin $|w| < \frac{s}{2}$ rounds directly to **0.0**.

### 3. Why Values Approach Zero (Signal Vanishing)
- **Mechanism:** When $s$ is inflated (e.g., driven by layer-wide outliers), the zero-bin widens. In deep layers with dense, small-magnitude distributions, 90–99% of weights fall into the zero-bin and round to 0.0.
- **Compounding Attenuation:** The effective layer gain drops below unity ($g \ll 1$). Across 18 layers, signal magnitude attenuates as $g^{18} \to 0$ (e.g. $0.6^{18} \approx 0.0001$).
- **ReLU Death:** When signals collapse toward zero, non-linear activations ($\text{ReLU}(x) = \max(0, x)$) permanently threshold neurons to zero, destroying all semantic discrimination and collapsing accuracy to random guessing (10.0%).

### 4. Why Values Blow Up (`inf` & `NaN`)
- **Mechanism:** If reconstructed weights are amplified by a scale factor $s$ that exceeds actual weight magnitudes (as observed when binarizing BatchNorm scales $\gamma$), each layer applies an artificial gain $g = \frac{s}{\text{mean}(|W|)} \gg 1$.
- **Overflowing Float32 Limits:** In naive 1-bit quantization, $g \approx 160\times$ per layer. Across 18 layers:
  $$\|x_{18}\| \approx 160^{18} \approx 1.2 \times 10^{39} > \text{float32}_{\max} \approx 3.4 \times 10^{38}$$
- **Result:** Activations overflow the single-precision ceiling into `+inf`. Downstream operations like normalization or $\text{Softmax}(\infty)$ produce `NaN` (Not-a-Number), crashing execution.

### 5. The Weight Clipping Dilemma (OMSE)
Uniform quantization faces a fundamental trade-off:
$$\text{Total Error} = \underbrace{\text{Clipping Error}}_{|w| > \alpha \text{ (tails truncated)}} + \underbrace{\text{Rounding Error}}_{|w| \le \alpha \text{ (grid step size } \Delta \propto \alpha)}$$
- **No Clipping ($\alpha = \max|W|$):** Large $\Delta$ causes rounding error to dominate, drowning weights into zero.
- **Excessive Clipping ($\alpha \ll \max|W|$):** Small $\Delta$ preserves dense resolution, but truncating tails causes signal attenuation.
- **Optimal MSE Clipping (OMSE):** Finds the optimal threshold $\alpha^* \in [0.6, 1.0]$ minimizing total $L_2$ distortion.

---

## The Socratic Learning Path

The project is structured into 5 progressive steps:

1. **Act I: The Free Lunch Illusion (Global INT8)**
   - *Question:* Can we simply round 32-bit floating point weights to 8-bit integers using a single scale factor?
   - *Result:* **97.8%** (matching FP32's 97.4%). Uniform 8-bit quantization is virtually lossless because 256 discrete bins keep rounding noise far below model tolerance.
2. **Act II: The Catastrophic Collapse (Global INT4 & The Outlier Tyranny)**
   - *Question:* If 8-bit worked effortlessly, why does Global INT4 collapse to **10.0%** (random guessing)?
   - *Investigation:* In `conv1`, high-magnitude outlier weights inflate the model-wide step size $\Delta = 0.145$. Because deep layers have mean magnitudes of only $0.015$, **98.4% of all weights in the network round directly to 0.0 (99.9% in `layer4`)**.
   - *The 1-Bit Overflow:* Binarizing BatchNorm scale $\gamma$ compounds a $160\times$ gain across 18 layers ($160^{18} \approx 10^{39} > \text{float32}_{\max}$), exploding to `inf` and `NaN`. Keeping BatchNorm in FP32 and applying mean-absolute scaling restores stability.
3. **Act III: Isolating Outliers (Per-Channel & OMSE Clipping)**
   - *Question:* If one outlier channel ruins the step size for the entire network, what if each output channel gets its own scale?
   - *Result:* PyTorch/TensorRT standard per-channel scaling recovers accuracy from 10.0% to **89.6%**.
   - *The Socratic Twist:* Can deliberately throwing away outlier tails *improve* accuracy?
   - *OMSE Clipping (Banner et al., NeurIPS 2019):* Saturating the outer 10–20% outlier tails ($\alpha = 0.8$) cuts the step size in half for the dense 80% bulk of weights, pushing accuracy to **93.0%**.
4. **Act IV: Sub-Channel Granularity (Group-Wise & Asymmetric Zero-Points)**
   - *Question:* In deep layers, a single channel has 4,608 weights. Is 1 scale per channel enough?
   - *Result:* Grouping weights into contiguous blocks of $G=32$ (AWQ / GPTQ / GGUF style) with asymmetric zero-points reaches **96.6%** at 4-bit, and recovers 3-bit accuracy from 82.6% (symmetric) to **91.8%** (asymmetric).
5. **Act V: The Architectural Discovery (Sensitivity-Guided Pareto Optimization)**
   - *Question:* Must every layer have the exact same bit-width?
   - *The Discovery:* Parameter distribution is severely skewed: `layer4` contains **75.1%** of all model weights, while sensitive bottleneck layers (`conv1` and skip projections) account for **$<6.2\%$**.
   - *The Breakthrough (HAWQ / Fisher style):* Allocating 8-bit or 4-bit to tiny sensitive layers adds $<0.05$ bits model-wide, allowing the 75% parameter bulk in `layer4` to be compressed to 2-bit or 3-bit:
     - **2.04 effective bits**: **92.8%** accuracy (4.85 MB)
     - **2.25 effective bits**: **95.6%** accuracy (5.14 MB)
     - **2.53 effective bits**: **97.8%** accuracy (**exceeds FP32 baseline** at an **$8.4\times$ compression ratio!**)

---

## How to Explore

### 1. Interactive Jupyter Notebook (Recommended)
Open and run the self-contained walkthrough notebook:
```bash
jupyter notebook notebooks/socratic_quantization_journey.ipynb
```
The notebook includes markdown discussions, diagnostic code cells, weight distribution histograms, zero-percentage layer bar charts, parameter distribution pie charts, Pareto curves, and an interactive custom bit budget allocator.

### 2. Step-by-Step CLI Walkthrough
Run any individual step or the entire progression from your terminal:
```bash
# Run step-by-step
python experiments/scripts/socratic_walkthrough.py --step 1
python experiments/scripts/socratic_walkthrough.py --step 2
python experiments/scripts/socratic_walkthrough.py --step 3
python experiments/scripts/socratic_walkthrough.py --step 4
python experiments/scripts/socratic_walkthrough.py --step 5

# Or run the complete story
python experiments/scripts/socratic_walkthrough.py --all
```

### 3. Full Benchmark Suite
To regenerate the full machine-readable benchmark JSON across all 21 evaluated configurations:
```bash
python experiments/scripts/run_quantization.py
```
Results are saved to `results/metrics/pretrained_resnet18_quantization.json`.

---

## Comprehensive Benchmark Comparison

Measured on a balanced 500-image sample (50 per class) from Imagenette's labeled validation split:

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

---

## Repository Layout

```text
notebooks/                   Interactive Socratic Jupyter Notebook
  socratic_quantization_journey.ipynb
experiments/
  configs/                   Run configurations
  scripts/                   CLI tools (socratic_walkthrough.py, run_quantization.py)
src/
  data/                      Validation and calibration data loaders
  evaluation/                Classification accuracy and loss metrics
  models/                    Pretrained model loaders
  quantization/integer/      Quantization algorithms (per-channel, group-wise, sensitivity-guided)
results/
  metrics/                   Machine-readable benchmark JSONs
```
