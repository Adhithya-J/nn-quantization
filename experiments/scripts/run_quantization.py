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
from quantization.integer.weight import quantize_activation_selected, quantize_weights


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
        result = {
            "metrics": metrics,
            "accuracy_drop": (results["fp32"]["accuracy"] - metrics["accuracy"]
                              if metrics["accuracy"] is not None else None),
            "nominal_weight_bits": bits,
            "estimated_storage_bytes": (
                sum(p.numel() for name, p in quantized.named_parameters() if name.endswith("weight")) * bits // 8
                + sum(p.numel() * 4 for name, p in quantized.named_parameters() if not name.endswith("weight"))
                + (4 if scales else 0)
            ),
            "quantization": scales,
        }
        results["methods"][f"int{bits}"] = result

    selective = copy.deepcopy(model)
    selective_metadata = quantize_activation_selected(
        selective, calibration_loader, device,
        max_images=config.get("calibration_images", 512),
    )
    selective_metrics = evaluate(selective, test_loader, device, class_indices)
    quantized_count = selective_metadata["quantized_weight_count"]
    selective_metadata["estimated_storage_bytes"] = (
        quantized_count * selective_metadata["effective_weight_bits"] / 8
        + (sum(parameter.numel() for parameter in selective.parameters()) - quantized_count) * 4
        + selective_metadata["weight_group_count"] * 4
    )
    results["methods"]["activation_selected_q4_mixed"] = {
        "metrics": selective_metrics,
        "accuracy_drop": (results["fp32"]["accuracy"] - selective_metrics["accuracy"]
                          if selective_metrics["accuracy"] is not None else None),
        "nominal_weight_bits": selective_metadata["effective_weight_bits"],
        "estimated_storage_bytes": selective_metadata["estimated_storage_bytes"],
        "quantization": selective_metadata,
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
