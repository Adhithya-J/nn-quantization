import torch
from torch import nn


@torch.no_grad()
def quantize_weights(model, bits):
    """Symmetric global-scale uniform weight quantization, dequantized to FP32.

    Conv and linear weights are quantized. Biases and batch-normalization parameters
    are left in FP32. One-bit uses the representable levels {-1, +1} with mean-absolute
    scaling (XNOR-Net style); other widths use signed two's-complement ranges scaled by
    maximum magnitude. Returns scale metadata by parameter name.
    """
    if bits not in (1, 2, 3, 4, 8, 16):
        raise ValueError("bits must be one of 1, 2, 3, 4, 8, or 16")
    metadata = {}
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and getattr(module, "weight", None) is not None:
            param_name = f"{name}.weight" if name else "weight"
            weights.append((param_name, module.weight))
    if not weights:
        return metadata
    if bits == 1:
        total_abs = sum(parameter.data.abs().sum().item() for _, parameter in weights)
        total_numel = sum(parameter.data.numel() for _, parameter in weights)
        scale = total_abs / total_numel if total_numel > 0 else 1.0
    else:
        qmax = 2 ** (bits - 1) - 1
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
def compute_channel_activity(model, calibration_loader, device, max_images=512):
    """Collect input-channel activity statistics from calibration images.

    Registers forward pre-hooks on Conv2d and Linear layers and forwards batches
    to record how frequently each input channel produces non-zero activations.
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

    return {
        "processed_images": processed,
        "scores": {name: active[name] / max(seen[name], 1) for name in modules},
        "mean_active": {name: float((active[name] / max(seen[name], 1)).mean()) for name in modules},
    }


@torch.no_grad()
def quantize_activation_selected(
    model,
    calibration_loader=None,
    device=None,
    low_bits=4,
    high_bits=6,
    high_ratio=0.25,
    max_images=512,
    activity_data=None,
):
    """Use input-channel activity to allocate mixed-precision weight groups.

    Conv/linear weights are grouped by input channel. Groups whose input channel
    is active more often get `high_bits`; the rest get `low_bits`.
    Each group has its own symmetric scale. Calibration images are forwarded
    through the original model to collect input activity statistics if `activity_data`
    is not provided.
    """
    if low_bits not in (1, 2, 4, 8, 16):
        raise ValueError(f"low_bits must be one of 1, 2, 4, 8, or 16, got {low_bits}")
    if high_bits not in (2, 4, 6, 8, 16, 32) or high_bits <= low_bits:
        raise ValueError(f"high_bits ({high_bits}) must be > low_bits ({low_bits}) and one of 2, 4, 6, 8, 16, 32")

    model.eval()
    modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }

    if activity_data is None:
        if calibration_loader is None or device is None:
            raise ValueError("Either activity_data or (calibration_loader and device) must be provided")
        activity_data = compute_channel_activity(model, calibration_loader, device, max_images=max_images)

    scores_dict = activity_data["scores"]
    processed = activity_data["processed_images"]

    groups = []
    for name, module in modules.items():
        weight = module.weight.data
        scores = scores_dict[name]
        if len(scores) != weight.shape[1]:
            raise ValueError(f"Activation/weight channel mismatch for layer {name}")
        for channel, score in enumerate(scores.tolist()):
            groups.append((float(score), name, channel, weight[:, channel, ...].numel()))

    total_weights = sum(group[3] for group in groups)
    high_bit_budget = int(total_weights * high_ratio)
    high_groups = set()
    allocated = 0
    for _, name, channel, size in sorted(groups, key=lambda group: (-group[0], group[1], group[2])):
        if allocated + size <= high_bit_budget:
            high_groups.add((name, channel))
            allocated += size

    details = {}
    for name, module in modules.items():
        details[name] = {
            "mean_active_fraction": float(scores_dict[name].mean()),
            "groups": 0,
            "high_precision_groups": 0,
        }
        for channel in range(module.weight.shape[1]):
            weight_group = module.weight.data[:, channel, ...]
            bits = high_bits if (name, channel) in high_groups else low_bits
            if bits == 32:
                # Kept unquantized in FP32
                pass
            elif bits == 1:
                scale = weight_group.abs().mean().item()
                scale = scale if scale > 0 else 1.0
                quantized = torch.where(weight_group >= 0, 1.0, -1.0)
                weight_group.copy_(quantized * scale)
            else:
                qmax = 2 ** (bits - 1) - 1
                qmin = -(2 ** (bits - 1))
                scale = weight_group.abs().max().item() / qmax
                scale = scale if scale > 0 else 1.0
                quantized = torch.clamp(torch.round(weight_group / scale), qmin, qmax)
                weight_group.copy_(quantized * scale)

            details[name]["groups"] += 1
            details[name]["high_precision_groups"] += int(bits == high_bits)

    quantized_weights = sum(group[3] for group in groups)
    high_weight_count = sum(size for _, name, channel, size in groups if (name, channel) in high_groups)
    effective_bits = (high_bits * high_weight_count + low_bits * (quantized_weights - high_weight_count)) / quantized_weights

    return {
        "scheme": f"activation_selected_{low_bits}_{high_bits}_bit_groups",
        "low_bits": low_bits,
        "high_bits": high_bits,
        "calibration_images": processed,
        "effective_weight_bits": effective_bits,
        "high_precision_weight_fraction": high_weight_count / quantized_weights,
        "quantized_weight_count": quantized_weights,
        "weight_group_count": len(groups),
        "layers": details,
    }


@torch.no_grad()
def quantize_per_channel(model, bits=4, mse_clipping=True, preserve_edges=None):
    """Symmetric per-output-channel weight quantization with optional MSE clipping.

    Standard deployment format in TensorRT, PyTorch, and ONNX Runtime.
    Quantizes Conv2d along dim 0 (per filter) and Linear along dim 0 (per row).
    If `mse_clipping` is True (Banner et al., NeurIPS 2019 / OMSE), searches for an
    optimal clipping threshold alpha in [0.6, 1.0] that minimizes L2 error ||W - W_q||^2.
    If `preserve_edges` is 'int8' or 'fp32', retains 'conv1' and 'fc' in higher precision.
    """
    if bits not in (1, 2, 3, 4, 8, 16):
        raise ValueError("bits must be one of 1, 2, 3, 4, 8, or 16")
    model.eval()
    scales_dict = {}
    total_quantized = 0
    total_weights = 0
    total_bits_sum = 0
    num_scales = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total_weights += w.numel()
            is_edge = name in ("conv1", "fc")
            layer_bits = bits
            if is_edge and preserve_edges == "fp32":
                total_bits_sum += 32 * w.numel()
                continue
            elif is_edge and preserve_edges == "int8":
                layer_bits = 8

            total_quantized += w.numel()
            total_bits_sum += layer_bits * w.numel()

            reduce_axes = tuple(range(1, w.ndim))
            num_channels = w.shape[0]
            num_scales += num_channels

            if layer_bits == 1:
                scale = w.abs().mean(dim=reduce_axes, keepdim=True)
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                quantized = torch.where(w >= 0, 1.0, -1.0)
                module.weight.data.copy_(quantized * scale)
            else:
                qmax = 2 ** (layer_bits - 1) - 1
                qmin = -(2 ** (layer_bits - 1))
                max_val = w.abs().amax(dim=reduce_axes, keepdim=True)

                if mse_clipping and layer_bits <= 4:
                    best_scale = max_val / qmax
                    best_scale = torch.where(best_scale > 0, best_scale, torch.ones_like(best_scale))
                    best_err = float("inf")
                    for alpha in (0.6, 0.7, 0.8, 0.9, 1.0):
                        cur_scale = (max_val * alpha) / qmax
                        cur_scale = torch.where(cur_scale > 0, cur_scale, torch.ones_like(cur_scale))
                        cur_q = torch.clamp(torch.round(w / cur_scale), qmin, qmax)
                        err = ((w - cur_q * cur_scale) ** 2).sum().item()
                        if err < best_err:
                            best_err = err
                            best_scale = cur_scale
                    scale = best_scale
                else:
                    scale = max_val / qmax
                    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

                quantized = torch.clamp(torch.round(w / scale), qmin, qmax)
                module.weight.data.copy_(quantized * scale)

            scales_dict[name] = {"channels": num_channels, "bits": layer_bits}

    eff_bits = total_bits_sum / total_weights if total_weights > 0 else bits
    return {
        "scheme": f"per_channel_int{bits}" + ("_mse_clipped" if mse_clipping else "") + (f"_preserve_{preserve_edges}" if preserve_edges else ""),
        "bits": bits,
        "effective_weight_bits": eff_bits,
        "quantized_weight_count": total_quantized,
        "scale_count": num_scales,
        "mse_clipping": mse_clipping,
        "preserve_edges": preserve_edges,
        "layers": scales_dict,
    }


def _quantize_tensor_groupwise(w, bits, group_size=32, asymmetric=False):
    """Internal helper to quantize a weight tensor in contiguous groups."""
    if bits >= 16:
        return w, 0
    flat_w = w.reshape(-1)
    n = flat_w.numel()
    num_groups = (n + group_size - 1) // group_size
    padded_len = num_groups * group_size
    if padded_len > n:
        padded = torch.zeros(padded_len, dtype=flat_w.dtype, device=flat_w.device)
        padded[:n] = flat_w
    else:
        padded = flat_w
    groups = padded.reshape(num_groups, group_size)

    if bits == 1:
        scale = groups.abs().mean(dim=1, keepdim=True)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        quantized = torch.where(groups >= 0, 1.0, -1.0)
        dequant = quantized * scale
    elif asymmetric:
        qmax = 2**bits - 1
        g_min = groups.amin(dim=1, keepdim=True)
        g_max = groups.amax(dim=1, keepdim=True)
        scale = (g_max - g_min) / qmax
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        zero_point = torch.round(-g_min / scale)
        q = torch.clamp(torch.round(groups / scale) + zero_point, 0, qmax)
        dequant = (q - zero_point) * scale
    else:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
        scale = groups.abs().amax(dim=1, keepdim=True) / qmax
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(groups / scale), qmin, qmax)
        dequant = q * scale

    return dequant.reshape(-1)[:n].reshape(w.shape), num_groups


@torch.no_grad()
def quantize_groupwise(model, bits=4, group_size=32, asymmetric=False):
    """Group-wise (block-wise) weight quantization (symmetric or asymmetric).

    Divides weights into contiguous blocks of `group_size` elements (standard
    in AWQ, GPTQ, and GGUF block quantization). Each group gets its own scale
    (and zero-point if asymmetric=True).
    """
    if bits not in (1, 2, 3, 4, 5, 6, 8, 16):
        raise ValueError(f"bits must be one of 1, 2, 3, 4, 5, 6, 8, 16, got {bits}")
    model.eval()
    total_quantized = 0
    total_scales = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            dequant, num_groups = _quantize_tensor_groupwise(
                w, bits=bits, group_size=group_size, asymmetric=asymmetric
            )
            module.weight.data.copy_(dequant)
            total_quantized += w.numel()
            total_scales += num_groups

    return {
        "scheme": f"groupwise_{'asym_' if asymmetric else ''}int{bits}_g{group_size}",
        "bits": bits,
        "group_size": group_size,
        "asymmetric": asymmetric,
        "effective_weight_bits": bits,
        "quantized_weight_count": total_quantized,
        "scale_count": total_scales,
    }


@torch.no_grad()
def compute_layer_sensitivities(
    model,
    calibration_loader,
    device,
    candidate_bits=(1, 2, 3, 4, 8),
    group_size=32,
    asymmetric=True,
    max_images=128,
):
    """Measures layer-wise quantization sensitivity (perturbation MSE) on calibration data.

    For each Conv2d and Linear module in the model, quantizes that single module
    to each candidate bit-width (leaving the rest of the network in FP32) and computes
    the Mean Squared Error (MSE) between the output logits of the perturbed model and
    the original FP32 model.

    This captures second-order sensitivity (HAWQ / Fisher style) without requiring
    expensive Hessian computation.
    """
    model.eval()
    calib_images = []
    for batch in calibration_loader:
        calib_images.append(batch[0].to(device))
        if sum(len(x) for x in calib_images) >= max_images:
            break
    if not calib_images:
        raise ValueError("Calibration loader produced no images")
    calib_images = torch.cat(calib_images, dim=0)[:max_images]

    fp32_output = model(calib_images)

    modules = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    sensitivity_data = []
    for name, module in modules:
        numel = module.weight.numel()
        errors = {}
        for b in candidate_bits:
            if b >= 16:
                errors[b] = 0.0
                continue
            original_weight = module.weight.data.clone()
            q_w, _ = _quantize_tensor_groupwise(
                original_weight, bits=b, group_size=group_size, asymmetric=asymmetric
            )
            module.weight.data.copy_(q_w)
            perturbed_output = model(calib_images)
            mse = ((fp32_output - perturbed_output) ** 2).mean().item()
            errors[b] = mse
            module.weight.data.copy_(original_weight)

        sensitivity_data.append({
            "name": name,
            "numel": numel,
            "errors": errors,
        })

    return {
        "candidate_bits": list(candidate_bits),
        "group_size": group_size,
        "asymmetric": asymmetric,
        "max_images": len(calib_images),
        "layers": sensitivity_data,
    }


@torch.no_grad()
def quantize_sensitivity_guided(
    model,
    target_effective_bits=2.5,
    calibration_loader=None,
    device=None,
    sensitivities=None,
    group_size=32,
    asymmetric=True,
    candidate_bits=(1, 2, 3, 4, 8),
    max_images=128,
):
    """Sensitivity-guided mixed-precision quantization maximizing accuracy under a bit budget.

    Uses precomputed or newly collected layer-wise perturbation sensitivities (HAWQ / Fisher style)
    to solve a marginal-gain optimization: sensitive layers (e.g. early convolutions, skip projections)
    receive higher precision (4 or 8 bits), while parameter-heavy, resilient layers receive lower precision
    (2 or 3 bits), achieving peak classification accuracy with minimum effective bits.
    """
    model.eval()
    if sensitivities is None:
        if calibration_loader is None or device is None:
            raise ValueError("Either sensitivities or (calibration_loader and device) must be provided")
        sensitivities = compute_layer_sensitivities(
            model,
            calibration_loader,
            device,
            candidate_bits=candidate_bits,
            group_size=group_size,
            asymmetric=asymmetric,
            max_images=max_images,
        )

    layers_info = sensitivities["layers"]
    total_weights = sum(item["numel"] for item in layers_info)
    target_total_bits = target_effective_bits * total_weights

    # Greedy marginal gain solver
    min_b = min(candidate_bits)
    alloc = {item["name"]: min_b for item in layers_info}

    def current_total_bits():
        return sum(alloc[item["name"]] * item["numel"] for item in layers_info)

    while True:
        best_gain = -1.0
        best_layer = None
        best_next_bit = None

        curr_bits = current_total_bits()
        for item in layers_info:
            name = item["name"]
            cur_b = alloc[name]
            higher_bits = [b for b in candidate_bits if b > cur_b]
            if not higher_bits:
                continue
            next_b = min(higher_bits)
            delta_bits = (next_b - cur_b) * item["numel"]
            if curr_bits + delta_bits > target_total_bits + 0.05 * total_weights:
                continue
            err_curr = item["errors"].get(cur_b, 0.0)
            err_next = item["errors"].get(next_b, 0.0)
            gain = (err_curr - err_next) / max(delta_bits, 1)
            if gain > best_gain:
                best_gain = gain
                best_layer = name
                best_next_bit = next_b

        if best_layer is None or best_gain <= 0:
            break
        alloc[best_layer] = best_next_bit

    # Apply quantization to model
    modules_dict = dict(model.named_modules())
    total_scales = 0
    actual_total_bits = 0
    layer_details = {}

    for item in layers_info:
        name = item["name"]
        numel = item["numel"]
        assigned_bits = alloc[name]
        actual_total_bits += assigned_bits * numel
        w = modules_dict[name].weight.data
        dequant, num_groups = _quantize_tensor_groupwise(
            w, bits=assigned_bits, group_size=group_size, asymmetric=asymmetric
        )
        modules_dict[name].weight.data.copy_(dequant)
        total_scales += num_groups
        layer_details[name] = {
            "bits": assigned_bits,
            "weights": numel,
            "weight_fraction": numel / total_weights,
            "groups": num_groups,
            "sensitivity_error": item["errors"].get(assigned_bits, 0.0),
        }

    effective_bits = actual_total_bits / total_weights

    return {
        "scheme": f"sensitivity_guided_g{group_size}_{'asym' if asymmetric else 'sym'}",
        "target_effective_bits": target_effective_bits,
        "effective_weight_bits": effective_bits,
        "group_size": group_size,
        "asymmetric": asymmetric,
        "quantized_weight_count": total_weights,
        "scale_count": total_scales,
        "layer_allocations": alloc,
        "layers": layer_details,
    }
