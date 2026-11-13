from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights, resnet18


PRETRAINED_MODELS = {"resnet18": ResNet18_Weights.DEFAULT}


def load_pretrained_model(name, checkpoint_path=None):
    """Load a standard pretrained model, caching the original state locally."""
    try:
        weights = PRETRAINED_MODELS[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported pretrained model {name!r}; available: {', '.join(PRETRAINED_MODELS)}"
        ) from exc
    if name.lower() == "resnet18":
        model = resnet18(weights=None)
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_path and checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        else:
            state = weights.get_state_dict(progress=True, check_hash=True)
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state, checkpoint_path)
        model.load_state_dict(state)
        return model, weights
    raise AssertionError("Registered model has no constructor")
