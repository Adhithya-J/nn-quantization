import torch
from torch import nn


def evaluate(model, loader, device, class_indices=None):
    model.eval()
    correct = total = loss_sum = nonfinite_samples = samples_seen = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            samples_seen += labels.numel()
            logits = model(inputs)
            if class_indices is not None:
                logits = logits[:, class_indices]
            finite = torch.isfinite(logits).all(dim=1)
            nonfinite_samples += (~finite).sum().item()
            if finite.any():
                valid_logits, valid_labels = logits[finite], labels[finite]
                loss_sum += criterion(valid_logits, valid_labels).item()
                correct += (valid_logits.argmax(1) == valid_labels).sum().item()
                total += valid_labels.numel()
    if samples_seen == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return {
        "loss": loss_sum / total if total else None,
        "accuracy": correct / total if total else None,
        "nonfinite_samples": nonfinite_samples,
    }
