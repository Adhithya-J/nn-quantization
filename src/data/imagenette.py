"""Download and load Imagenette's labeled validation split for quick testing."""
import hashlib
import random
import tarfile
import urllib.request
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder


URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
MD5 = "e793b78cc4c9e9a4ccc0c1155377a412"
ARCHIVE_NAME = "imagenette2-160.tgz"


def _download_and_extract(root):
    root = Path(root)
    extracted = root / "imagenette2-160"
    if (extracted / "val").is_dir():
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE_NAME
    def archive_md5():
        digest = hashlib.md5()
        with archive.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    if not archive.is_file() or archive_md5() != MD5:
        request = urllib.request.Request(URL, headers={"User-Agent": "nn-quantization-study/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if archive_md5() != MD5:
            archive.unlink(missing_ok=True)
            raise RuntimeError("Downloaded Imagenette archive failed its MD5 integrity check")

    root_resolved = root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root_resolved) or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsafe path or file type in Imagenette archive: {member.name}")
        bundle.extractall(root, members=members)
    if not (extracted / "val").is_dir():
        raise RuntimeError("Imagenette archive did not contain the expected val/ directory")
    return extracted


def make_test_loader(root, weights, batch_size=64, num_workers=0,
                     images_per_class=50, seed=42):
    """Load Imagenette validation images with labels matching ImageNet-1K logits."""
    dataset_root = _download_and_extract(root)
    dataset = ImageFolder(dataset_root / "val", transform=weights.transforms())
    imagenette_to_imagenet = {
        "n01440764": "tench",
        "n02102040": "English springer",
        "n02979186": "cassette player",
        "n03000684": "chain saw",
        "n03028079": "church",
        "n03394916": "French horn",
        "n03417042": "garbage truck",
        "n03425413": "gas pump",
        "n03445777": "golf ball",
        "n03888257": "parachute",
    }
    categories = weights.meta["categories"]
    class_index = {name: index for index, name in enumerate(categories)}
    if set(dataset.classes) != set(imagenette_to_imagenet):
        raise RuntimeError(f"Unexpected Imagenette classes: {dataset.classes}")
    missing_labels = set(imagenette_to_imagenet.values()) - set(class_index)
    if missing_labels:
        raise RuntimeError(f"Model weights do not define Imagenette labels: {sorted(missing_labels)}")
    ordered_folders = sorted(imagenette_to_imagenet)
    label_by_folder = {folder: index for index, folder in enumerate(ordered_folders)}
    dataset.samples = [
        (path, label_by_folder[Path(path).parent.name]) for path, _ in dataset.samples
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    if images_per_class:
        rng = random.Random(seed)
        indices_by_class = {index: [] for index in range(len(ordered_folders))}
        for index, target in enumerate(dataset.targets):
            indices_by_class[target].append(index)
        selected = []
        for target in range(len(ordered_folders)):
            available = indices_by_class[target]
            if len(available) < images_per_class:
                raise ValueError(
                    f"Class {ordered_folders[target]} has only {len(available)} test images"
                )
            selected.extend(rng.sample(available, images_per_class))
        dataset = Subset(dataset, selected)
    model_class_indices = [
        class_index[imagenette_to_imagenet[folder]] for folder in ordered_folders
    ]
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        model_class_indices,
    )


def make_calibration_loader(root, weights, batch_size=64, num_workers=0, max_images=512):
    """Load training images only for activation statistics (never for scoring)."""
    dataset_root = _download_and_extract(root)
    dataset = ImageFolder(dataset_root / "train", transform=weights.transforms())
    if max_images and max_images < len(dataset):
        dataset = Subset(dataset, range(max_images))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
