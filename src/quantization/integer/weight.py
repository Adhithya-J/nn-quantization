import torch
from torch import nn


@torch.no_grad()
def quantize_weights(model, bits):
    """Symmetric global-scale uniform weight quantization, dequantized to FP32.

    Scales each parameter tensor independently. Biases are left in FP32.
    One-bit uses the representable levels {-1, +1}; other widths use signed
    two's-complement ranges. Returns scale metadata by parameter name.
    """
    if bits not in (1, 2, 4, 8, 16):
        raise ValueError("bits must be one of 1, 2, 4, 8, or 16")
    metadata = {}
    weights = [(name, parameter) for name, parameter in model.named_parameters()
               if name.endswith("weight")]
    if not weights:
        return metadata
    qmax = 1 if bits == 1 else 2 ** (bits - 1) - 1
    global_max = max(parameter.data.abs().max().item() for _, parameter in weights)
    scale = global_max / qmax
    scale = scale if scale > 0 else 1.0
    for name, parameter in weights:
        weight = parameter.data
        if bits == 1:
            quantized = torch.where(weight >= 0, 1.0, -1.0)
        else:
            qmin = -(2 ** (bits - 1))
            quantized = torch.clamp(torch.round(weight / scale), qmin, qmax)
        parameter.copy_(quantized * scale)
        metadata[name] = {"scale": scale, "bits": bits, "scheme": "symmetric_global"}
    return {
        "scheme": "symmetric_global",
        "bits": bits,
        "scale": scale,
        "quantized_tensors": [name for name, _ in weights],
    }


@torch.no_grad()
def quantize_activation_selected(model, calibration_loader, device, max_images=512):
    """Use input-channel activity to allocate 2/6-bit weight groups at ~4-bit average.

    Conv/linear weights are grouped by input channel. Groups whose input channel
    is active more often get 6 bits; the rest get 4 bits, for about 4.5 bits per
    weight. Each group has its own symmetric scale. Calibration images are forwarded through the original
    model only to collect input activity statistics.
    """
    model.eval()
    modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }
    active = {name: None for name in modules}
    seen = {name: 0 for name in modules}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs):
            values = inputs[0].detach()
            channel_axis = 1
            reduce_axes = tuple(axis for axis in range(values.ndim) if axis != channel_axis)
            channel_activity = (values.abs() > 1e-6).float().sum(dim=reduce_axes).cpu()
            count = 1
            for axis in reduce_axes:
                count *= values.shape[axis]
            active[name] = channel_activity if active[name] is None else active[name] + channel_activity
            seen[name] += count
        return hook

    for name, module in modules.items():
        hooks.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        processed = 0
        with torch.inference_mode():
            for batch in calibration_loader:
                images = batch[0].to(device)
                remaining = max_images - processed if max_images else len(images)
                if remaining <= 0:
                    break
                images = images[:remaining]
                model(images)
                processed += len(images)
    finally:
        for hook in hooks:
            hook.remove()
    if processed == 0:
        raise ValueError("Calibration loader produced no images")

    groups = []
    for name, module in modules.items():
        weight = module.weight.data
        scores = active[name] / max(seen[name], 1)
        if len(scores) != weight.shape[1]:
            raise ValueError(f"Activation/weight channel mismatch for layer {name}")
        for channel, score in enumerate(scores.tolist()):
            groups.append((float(score), name, channel, weight[:, channel, ...].numel()))

    total_weights = sum(group[3] for group in groups)
    high_bit_budget = total_weights // 4
    high_groups = set()
    allocated = 0
    for _, name, channel, size in sorted(groups, key=lambda group: (-group[0], group[1], group[2])):
        if allocated + size <= high_bit_budget:
            high_groups.add((name, channel))
            allocated += size

    details = {}
    for name, module in modules.items():
        details[name] = {"mean_active_fraction": float((active[name] / max(seen[name], 1)).mean()),
                         "groups": 0, "high_precision_groups": 0}
        for channel in range(module.weight.shape[1]):
            weight_group = module.weight.data[:, channel, ...]
            bits = 6 if (name, channel) in high_groups else 4
            qmax = 2 ** (bits - 1) - 1
            scale = weight_group.abs().max().item() / qmax
            scale = scale if scale > 0 else 1.0
            quantized = torch.clamp(torch.round(weight_group / scale), -qmax, qmax)
            weight_group.copy_(quantized * scale)
            details[name]["groups"] += 1
            details[name]["high_precision_groups"] += int(bits == 6)
    quantized_weights = sum(group[3] for group in groups)
    high_weight_count = sum(size for _, name, channel, size in groups if (name, channel) in high_groups)
    effective_bits = (6 * high_weight_count + 4 * (quantized_weights - high_weight_count)) / quantized_weights
    return {
        "scheme": "activation_selected_2_6_bit_groups",
        "calibration_images": processed,
        "effective_weight_bits": effective_bits,
        "high_precision_weight_fraction": high_weight_count / quantized_weights,
        "quantized_weight_count": quantized_weights,
        "weight_group_count": len(groups),
        "layers": details,
    }
