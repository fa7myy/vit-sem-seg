#!/usr/bin/env python3
"""
VOC transforms and dataloader construction.
"""

from __future__ import annotations

import csv
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

from .logging import RunLogger, log, write_run_config

try:
    from torchvision.datasets import VOCSegmentation
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception as exc:
    raise RuntimeError("torchvision is required for VOCSegmentation") from exc

from eval_utils import VOC_CLASS_NAMES


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_IGNORE_INDEX = 255


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    ignore_index: int
    class_names: Tuple[str, ...] | None = None

def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class IndexMaskTransform:
    def __init__(self, size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.size = (size, size)
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        target = TF.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        target = torch.from_numpy(np.array(target, dtype="uint8")).long()
        return image, target


class RgbMaskTransform:
    def __init__(
        self,
        size: int,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        rgb_to_id: Dict[int, int],
        ignore_index: int,
    ):
        self.size = (size, size)
        self.mean = mean
        self.std = std
        self.rgb_to_id = dict(rgb_to_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        target = TF.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        rgb = np.array(target, dtype="uint8")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB target mask, got shape={rgb.shape}")
        keys = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2].astype(np.int32)
        labels = np.full(keys.shape, self.ignore_index, dtype=np.int64)
        for key, class_id in self.rgb_to_id.items():
            labels[keys == int(key)] = int(class_id)
        target_tensor = torch.from_numpy(labels).long()
        return image, target_tensor


def resolve_input_norm(args: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if args.input_norm == "clip":
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def _resolve_voc_root(data_root: Path) -> Path:
    candidate = data_root / "VOC2012"
    if candidate.exists():
        return candidate
    return data_root


def _read_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in text if line.strip()]


def _voc_split_file(voc_root: Path, split: str) -> Path:
    flat = voc_root / f"{split}.txt"
    if flat.exists():
        return flat
    nested = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if nested.exists():
        return nested
    return flat


class VocFileDataset(Dataset):
    def __init__(self, ids: List[str], images_dir: Path, masks_dir: Path, transforms):
        self.ids = list(ids)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        img_path = self.images_dir / f"{image_id}.jpg"
        mask_path = self.masks_dir / f"{image_id}.png"
        image = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path)
        return self.transforms(image, target)


def _build_voc_datasets(args: Any, transforms) -> Tuple[Dataset, Dataset, DatasetSpec, Dict[str, Any]]:
    if args.voc_train_split == "trainaug" and args.download:
        raise ValueError("--download does not support VOC trainaug (SegmentationClassAug must exist locally).")

    voc_root = _resolve_voc_root(Path(args.data_root))
    images_dir = voc_root / "JPEGImages"
    masks_val_dir = voc_root / "SegmentationClass"
    masks_train_dir = masks_val_dir
    train_split = "train"
    if args.voc_train_split == "trainaug":
        train_split = "trainaug"
        masks_train_dir = voc_root / str(args.voc_aug_mask_dir)

    split_path = _voc_split_file(voc_root, train_split)
    val_split_path = _voc_split_file(voc_root, "val")
    if split_path.exists() and val_split_path.exists() and images_dir.exists() and masks_val_dir.exists():
        if args.voc_train_split == "trainaug" and not masks_train_dir.exists():
            raise FileNotFoundError(f"VOC trainaug masks dir not found: {masks_train_dir}")
        train_ids = _read_lines(split_path)
        val_ids = _read_lines(val_split_path)
        train_ds = VocFileDataset(train_ids, images_dir=images_dir, masks_dir=masks_train_dir, transforms=transforms)
        val_ds = VocFileDataset(val_ids, images_dir=images_dir, masks_dir=masks_val_dir, transforms=transforms)
        spec = DatasetSpec(name="voc", num_classes=21, ignore_index=DEFAULT_IGNORE_INDEX, class_names=VOC_CLASS_NAMES)
        meta = {
            "dataset": "voc",
            "voc_root": str(voc_root),
            "train_split": train_split,
            "train_list": str(split_path),
            "val_list": str(val_split_path),
            "train_masks_dir": str(masks_train_dir),
            "val_masks_dir": str(masks_val_dir),
        }
        return train_ds, val_ds, spec, meta

    # Fallback to torchvision (supports download) for standard VOC layout.
    try:
        train_ds = VOCSegmentation(
            root=str(Path(args.data_root)),
            year="2012",
            image_set="train",
            download=bool(args.download),
            transforms=transforms,
        )
        val_ds = VOCSegmentation(
            root=str(Path(args.data_root)),
            year="2012",
            image_set="val",
            download=bool(args.download),
            transforms=transforms,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"VOC load failed at data_root='{args.data_root}'. "
            "Expected layout: <data_root>/VOC2012 with subfolders JPEGImages, SegmentationClass, "
            "ImageSets/Segmentation, etc. If VOC2012 lives elsewhere, point --data-root to its parent. "
            "Original error: " + str(error)
        ) from error
    spec = DatasetSpec(name="voc", num_classes=21, ignore_index=DEFAULT_IGNORE_INDEX, class_names=VOC_CLASS_NAMES)
    meta = {"dataset": "voc", "train_split": "train", "source": "torchvision.VOCSegmentation"}
    return train_ds, val_ds, spec, meta


def _resolve_camvid_root(data_root: Path) -> Path:
    if (data_root / "train").exists() and (data_root / "train_labels").exists():
        return data_root
    candidate = data_root / "CamVid"
    if (candidate / "train").exists() and (candidate / "train_labels").exists():
        return candidate
    return data_root


def _parse_camvid_ignore_names(raw: str) -> set:
    parts = [p.strip().lower() for p in (raw or "").split(",")]
    return {p for p in parts if p}


def _load_camvid_class_dict(path: Path, ignore_names: set) -> Tuple[DatasetSpec, Dict[int, int], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"CamVid class_dict.csv not found: {path}")
    rgb_to_id: Dict[int, int] = {}
    class_names: List[str] = []
    ignored_rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"name", "r", "g", "b"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CamVid class_dict.csv must have columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            name = str(row["name"])
            r = int(row["r"])
            g = int(row["g"])
            b = int(row["b"])
            key = (r << 16) | (g << 8) | b
            if name.strip().lower() in ignore_names:
                rgb_to_id[key] = DEFAULT_IGNORE_INDEX
                ignored_rows.append({"name": name, "r": r, "g": g, "b": b})
                continue
            class_id = len(class_names)
            class_names.append(name)
            rgb_to_id[key] = class_id
    spec = DatasetSpec(name="camvid", num_classes=len(class_names), ignore_index=DEFAULT_IGNORE_INDEX, class_names=tuple(class_names))
    meta = {
        "dataset": "camvid",
        "class_dict_csv": str(path),
        "ignored_classes": ignored_rows,
        "num_classes": spec.num_classes,
        "ignore_index": spec.ignore_index,
    }
    return spec, rgb_to_id, meta


def _list_images(folder: Path) -> List[Path]:
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(folder.glob(ext))
    return sorted(paths)


def _pair_camvid_images_labels(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    images = _list_images(images_dir)
    labels = _list_images(labels_dir)
    img_by_stem = {p.stem: p for p in images}
    pairs: List[Tuple[Path, Path]] = []
    missing: List[str] = []
    for lab in labels:
        stem = lab.stem
        candidates = [stem]
        if stem.endswith("_L"):
            candidates.insert(0, stem[:-2])
        img_path = None
        for c in candidates:
            if c in img_by_stem:
                img_path = img_by_stem[c]
                break
        if img_path is None:
            missing.append(stem)
            continue
        pairs.append((img_path, lab))
    if missing:
        raise FileNotFoundError(f"CamVid: could not match {len(missing)} label(s) to images. Example: {missing[0]}")
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], transforms):
        self.pairs = list(pairs)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, lab_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        target = Image.open(lab_path).convert("RGB")
        return self.transforms(image, target)


def _build_camvid_datasets(args: Any, transforms) -> Tuple[Dataset, Dataset, DatasetSpec, Dict[str, Any]]:
    camvid_root = _resolve_camvid_root(Path(args.data_root))
    ignore_names = _parse_camvid_ignore_names(getattr(args, "camvid_ignore_class_names", ""))
    spec, rgb_to_id, meta = _load_camvid_class_dict(camvid_root / "class_dict.csv", ignore_names=ignore_names)
    # Override transforms to include RGB decoding.
    image_mean, image_std = resolve_input_norm(args)
    camvid_tf = RgbMaskTransform(
        size=args.img_size,
        mean=image_mean,
        std=image_std,
        rgb_to_id=rgb_to_id,
        ignore_index=spec.ignore_index,
    )
    train_pairs = _pair_camvid_images_labels(camvid_root / "train", camvid_root / "train_labels")
    val_pairs = _pair_camvid_images_labels(camvid_root / "val", camvid_root / "val_labels")
    meta.update(
        {
            "camvid_root": str(camvid_root),
            "train_images_dir": str(camvid_root / "train"),
            "train_labels_dir": str(camvid_root / "train_labels"),
            "val_images_dir": str(camvid_root / "val"),
            "val_labels_dir": str(camvid_root / "val_labels"),
        }
    )
    return PairDataset(train_pairs, camvid_tf), PairDataset(val_pairs, camvid_tf), spec, meta


def _apply_train_subset(
    args: Any,
    train_set: Dataset,
    run_logger: RunLogger | None,
    split_percent: int | None,
    split_seed: int | None,
) -> Tuple[Dataset, int, Dict[str, Any] | None]:
    full_train_size = len(train_set)
    subset_info = None
    if split_percent is not None:
        if split_percent <= 0 or split_percent > 100:
            raise ValueError(f"split_percent must be in 1..100, got {split_percent}")
        effective_seed = int(args.seed) if split_seed is None else int(split_seed)
        if split_percent == 100:
            subset_size = full_train_size
        else:
            subset_size = max(1, int(round(full_train_size * (split_percent / 100.0))))
        gen = torch.Generator()
        gen.manual_seed(effective_seed)
        perm = torch.randperm(full_train_size, generator=gen).tolist()
        subset_indices = perm[:subset_size]
        sha1 = hashlib.sha1((",".join(str(i) for i in subset_indices)).encode("utf-8")).hexdigest()
        realized_percent = 100.0 * (len(subset_indices) / float(full_train_size)) if full_train_size > 0 else 0.0
        subset_info = {
            "requested_percent": int(split_percent),
            "realized_percent": realized_percent,
            "subset_size": int(len(subset_indices)),
            "full_train_size": int(full_train_size),
            "split_seed": int(effective_seed),
            "selection": "torch.randperm_prefix",
            "indices_sha1": sha1,
        }
        train_set = Subset(train_set, subset_indices)
        if run_logger is not None:
            run_logger.write_json(
                "train_subset_indices.json",
                {
                    "split": subset_info,
                    "indices": subset_indices,
                },
            )
    return train_set, full_train_size, subset_info


def build_loaders(
    args: Any,
    run_logger: RunLogger | None,
    run_info: Dict[str, Any],
    split_percent: int | None = None,
    split_seed: int | None = None,
) -> Tuple[DataLoader, DataLoader, DatasetSpec]:
    if not args.data_root:
        raise ValueError("--data-root is required unless --dry-run is set.")

    image_mean, image_std = resolve_input_norm(args)
    log(f"[data] dataset={args.dataset} input_norm={args.input_norm}", run_logger)

    if args.dataset == "voc":
        transforms = IndexMaskTransform(args.img_size, mean=image_mean, std=image_std)
        train_set, val_set, spec, meta = _build_voc_datasets(args, transforms)
    elif args.dataset == "camvid":
        # transforms for camvid are built inside _build_camvid_datasets (need class_dict mapping).
        train_set, val_set, spec, meta = _build_camvid_datasets(args, transforms=None)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_set, full_train_size, subset_info = _apply_train_subset(args, train_set, run_logger, split_percent, split_seed)
    # Log before training starts so sweep runs clearly show the dataset fraction and absolute size.
    if subset_info is None:
        log(f"[data] train_subset=100% train_images={len(train_set)}/{int(full_train_size)}", run_logger)
    else:
        log(
            "[data] train_subset={requested}% (realized={realized:.2f}%) train_images={n}/{full} split_seed={seed}".format(
                requested=int(subset_info["requested_percent"]),
                realized=float(subset_info["realized_percent"]),
                n=int(subset_info["subset_size"]),
                full=int(subset_info["full_train_size"]),
                seed=int(subset_info["split_seed"]),
            ),
            run_logger,
        )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    run_info["dataset"] = {
        "name": spec.name,
        "train_size": len(train_set),
        "full_train_size": int(full_train_size),
        "val_size": len(val_set),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "meta": meta,
    }
    run_info["dataset_spec"] = {
        "name": spec.name,
        "num_classes": spec.num_classes,
        "ignore_index": spec.ignore_index,
        "class_names": list(spec.class_names) if spec.class_names is not None else None,
    }
    if subset_info is not None:
        run_info["dataset"]["train_subset"] = subset_info
    write_run_config(run_logger, run_info)
    return train_loader, val_loader, spec


def get_dataset_spec(args: Any) -> DatasetSpec:
    if getattr(args, "dataset", "voc") == "voc":
        return DatasetSpec(name="voc", num_classes=21, ignore_index=DEFAULT_IGNORE_INDEX, class_names=VOC_CLASS_NAMES)
    if getattr(args, "dataset", "") == "camvid":
        if not getattr(args, "data_root", ""):
            raise ValueError("--data-root is required for camvid.")
        camvid_root = _resolve_camvid_root(Path(args.data_root))
        ignore_names = _parse_camvid_ignore_names(getattr(args, "camvid_ignore_class_names", ""))
        spec, _rgb_to_id, _meta = _load_camvid_class_dict(camvid_root / "class_dict.csv", ignore_names=ignore_names)
        return spec
    raise ValueError(f"Unknown dataset: {getattr(args, 'dataset', None)}")


def build_voc_loaders(
    args: Any,
    run_logger: RunLogger | None,
    run_info: Dict[str, Any],
    split_percent: int | None = None,
    split_seed: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    train_loader, val_loader, _spec = build_loaders(args, run_logger, run_info, split_percent, split_seed)
    return train_loader, val_loader
