import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.imagenette import make_calibration_loader, make_test_loader
from evaluation.metrics import evaluate
from models.vision import load_pretrained_model
from quantization.integer.weight import (
    compute_channel_activity,
    compute_layer_sensitivities,
    quantize_activation_selected,
    quantize_groupwise,
    quantize_per_channel,
    quantize_sensitivity_guided,
    quantize_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Quantize an internet-hosted pretrained vision model")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/pretrained_resnet18.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep both torchvision's download cache and an explicit original checkpoint
    # inside the project so later runs do not need internet access.
    torch.hub.set_dir(str(ROOT / config["data_root"] / "torch_cache"))
    checkpoint_path = ROOT / config["output_dir"] / "checkpoints" / f"{config['model']}_pretrained_fp32.pt"
    model, weights = load_pretrained_model(config["model"], checkpoint_path)
    model = model.to(device).eval()
    test_loader, class_indices = make_test_loader(
        ROOT / config["data_root"], weights,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        images_per_class=config.get("test_images_per_class", 50),
        seed=config.get("seed", 42),
    )
    calibration_loader = make_calibration_loader(
        ROOT / config["data_root"], weights,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 0),
        max_images=config.get("calibration_images", 512),
    )

    output = ROOT / config["output_dir"]
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    results = {
        "model": config["model"],
        "weights": str(weights),
        "device": str(device),
        "parameter_count": parameter_count,
        "test_dataset": config["test_dataset"],
        "test_images": len(test_loader.dataset),
        "calibration_dataset": "imagenette/train",
        "calibration_images": config.get("calibration_images", 512),
        "original_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "fp32_storage_bytes": parameter_count * 4,
        "evaluation_classes": [weights.meta["categories"][index] for index in class_indices],
        "fp32": evaluate(model, test_loader, device, class_indices),
        "methods": {},
    }

    for bits in config["bits"]:
        quantized = copy.deepcopy(model)
        scales = quantize_weights(quantized, bits)
        metrics = evaluate(quantized, test_loader, device, class_indices)
        quantized_names = set(scales["quantized_tensors"]) if scales else set()
        result = {
            "metrics": metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - metrics["accuracy"]
                              if metrics["accuracy"] is not None else None),
            "nominal_weight_bits": bits,
            "estimated_storage_bytes": (
                sum(p.numel() * bits // 8 for name, p in quantized.named_parameters() if name in quantized_names)
                + sum(p.numel() * 4 for name, p in quantized.named_parameters() if name not in quantized_names)
                + (4 if scales else 0)
            ),
            "quantization": scales,
        }
        results["methods"][f"int{bits}"] = result

    activity_data = compute_channel_activity(
        model, calibration_loader, device,
        max_images=config.get("calibration_images", 512),
    )

    mixed_pairs = [
        (1, 2),
        (2, 4),
        (4, 6),
        (4, 8),
        (8, 16),
        (16, 32),
    ]

    for low_bits, high_bits in mixed_pairs:
        selective = copy.deepcopy(model)
        selective_metadata = quantize_activation_selected(
            selective,
            activity_data=activity_data,
            low_bits=low_bits,
            high_bits=high_bits,
        )
        selective_metrics = evaluate(selective, test_loader, device, class_indices)
        quantized_count = selective_metadata["quantized_weight_count"]
        storage_bytes = (
            quantized_count * selective_metadata["effective_weight_bits"] / 8
            + (sum(parameter.numel() for parameter in selective.parameters()) - quantized_count) * 4
            + selective_metadata["weight_group_count"] * 4
        )
        selective_metadata["estimated_storage_bytes"] = storage_bytes
        method_name = f"activation_selected_q{low_bits}_q{high_bits}"
        results["methods"][method_name] = {
            "metrics": selective_metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - selective_metrics["accuracy"]
                              if selective_metrics["accuracy"] is not None else None),
            "nominal_weight_bits": selective_metadata["effective_weight_bits"],
            "estimated_storage_bytes": storage_bytes,
            "quantization": selective_metadata,
        }
        if (low_bits, high_bits) == (4, 6):
            results["methods"]["activation_selected_q4_mixed"] = results["methods"][method_name]

    # Modern Literature Methods:
    # 1. Per-Channel Quantization & MSE Optimal Clipping (Banner et al., NeurIPS 2019)
    per_channel_configs = [
        ("per_channel_int8", 8, False, None),
        ("per_channel_int4", 4, False, None),
        ("per_channel_int4_omse", 4, True, None),
        ("per_channel_int4_omse_preserve_int8", 4, True, "int8"),
    ]
    for name, bits, mse, preserve in per_channel_configs:
        pc_model = copy.deepcopy(model)
        pc_meta = quantize_per_channel(pc_model, bits=bits, mse_clipping=mse, preserve_edges=preserve)
        pc_metrics = evaluate(pc_model, test_loader, device, class_indices)
        quantized_count = pc_meta["quantized_weight_count"]
        storage_bytes = (
            quantized_count * pc_meta["effective_weight_bits"] / 8
            + (sum(p.numel() for p in pc_model.parameters()) - quantized_count) * 4
            + pc_meta["scale_count"] * 4
        )
        pc_meta["estimated_storage_bytes"] = storage_bytes
        results["methods"][name] = {
            "metrics": pc_metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - pc_metrics["accuracy"]
                              if pc_metrics["accuracy"] is not None else None),
            "nominal_weight_bits": pc_meta["effective_weight_bits"],
            "estimated_storage_bytes": storage_bytes,
            "quantization": pc_meta,
        }

    # 2. Group-wise / Block-wise Quantization (AWQ, GPTQ, GGUF style)
    groupwise_configs = [
        ("groupwise_int4_g64", 4, 64, False),
        ("groupwise_int4_g32", 4, 32, False),
        ("groupwise_int4_g32_asym", 4, 32, True),
        ("groupwise_int3_g32", 3, 32, False),
        ("groupwise_int3_g32_asym", 3, 32, True),
    ]
    for name, gbits, gsize, gasym in groupwise_configs:
        gw_model = copy.deepcopy(model)
        gw_meta = quantize_groupwise(gw_model, bits=gbits, group_size=gsize, asymmetric=gasym)
        gw_metrics = evaluate(gw_model, test_loader, device, class_indices)
        quantized_count = gw_meta["quantized_weight_count"]
        scale_bytes = gw_meta["scale_count"] * (5 if gasym else 4)  # 4B scale + 1B zero-point if asym
        storage_bytes = (
            quantized_count * gw_meta["effective_weight_bits"] / 8
            + (sum(p.numel() for p in gw_model.parameters()) - quantized_count) * 4
            + scale_bytes
        )
        gw_meta["estimated_storage_bytes"] = storage_bytes
        results["methods"][name] = {
            "metrics": gw_metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - gw_metrics["accuracy"]
                              if gw_metrics["accuracy"] is not None else None),
            "nominal_weight_bits": gw_meta["effective_weight_bits"],
            "estimated_storage_bytes": storage_bytes,
            "quantization": gw_meta,
        }

    # 3. Sensitivity-Guided Mixed-Precision Allocation (HAWQ / Fisher Pareto Optimization)
    print("Computing calibration layer perturbation sensitivities for Pareto optimization...")
    sensitivities = compute_layer_sensitivities(
        model,
        calibration_loader,
        device,
        candidate_bits=(1, 2, 3, 4, 8),
        group_size=32,
        asymmetric=True,
        max_images=128,
    )

    pareto_targets = [2.0, 2.2, 2.5, 3.0, 3.5, 4.0]
    for target_bits in pareto_targets:
        sg_model = copy.deepcopy(model)
        sg_meta = quantize_sensitivity_guided(
            sg_model,
            target_effective_bits=target_bits,
            sensitivities=sensitivities,
            group_size=32,
            asymmetric=True,
            candidate_bits=(1, 2, 3, 4, 8),
        )
        sg_metrics = evaluate(sg_model, test_loader, device, class_indices)
        quantized_count = sg_meta["quantized_weight_count"]
        scale_bytes = sg_meta["scale_count"] * 5  # 4B scale + 1B zero-point
        storage_bytes = (
            quantized_count * sg_meta["effective_weight_bits"] / 8
            + (sum(p.numel() for p in sg_model.parameters()) - quantized_count) * 4
            + scale_bytes
        )
        sg_meta["estimated_storage_bytes"] = storage_bytes
        method_name = f"pareto_sensitivity_guided_{target_bits}b"
        results["methods"][method_name] = {
            "metrics": sg_metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - sg_metrics["accuracy"]
                              if sg_metrics["accuracy"] is not None else None),
            "nominal_weight_bits": round(sg_meta["effective_weight_bits"], 2),
            "estimated_storage_bytes": storage_bytes,
            "quantization": sg_meta,
        }

    path = output / "metrics/pretrained_resnet18_quantization.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary = {"fp32": results["fp32"], "methods": {
        name: {"metrics": entry["metrics"], "accuracy_drop": entry["accuracy_drop"],
               "nominal_weight_bits": entry["nominal_weight_bits"],
               "estimated_storage_bytes": entry["estimated_storage_bytes"]}
        for name, entry in results["methods"].items()
    }}
    print(json.dumps(summary, indent=2))
    print(f"Saved metrics to {path}")


if __name__ == "__main__":
    main()
