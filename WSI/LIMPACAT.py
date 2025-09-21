# Inference-only script for MONAI MIL WSI classification
# Saves a CSV to ./results named "<experiment_name>.csv" with columns: path,label

import argparse
import os
import gc
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import re
from torch.utils.data.dataloader import default_collate
from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    MapTransform,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from torch.amp import autocast
import csv


def aggressive_memory_cleanup():
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


class LabelEncodeIntegerGraded(MapTransform):
    def __init__(self, num_classes: int, keys: KeysCollection = "label", allow_missing_keys: bool = True):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            try:
                label = int(d[key])
                lz = np.zeros(self.num_classes, dtype=np.float32)
                lz[:label] = 1.0
                d[key] = lz
            except Exception:
                pass
        return d


def list_data_collate(batch):
    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)


def run_inference(args):
    os.makedirs(args.results_dir, exist_ok=True)

    datalist = load_decathlon_datalist(
        data_list_file_path=args.dataset_json,
        data_list_key=args.split_key,
        base_dir=args.data_root,
    )
    if args.quick:
        datalist = datalist[:min(16, len(datalist))]
    if len(datalist) == 0:
        raise RuntimeError(f"No items found in split '{args.split_key}' of {args.dataset_json}")

    infer_transform = Compose([
        LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True),
        GridPatchd(
            keys=["image"],
            patch_size=(args.tile_size, args.tile_size),
            threshold=0.99 * 3 * 255 * args.tile_size * args.tile_size,
            pad_mode=None,
            constant_values=255,
        ),
        SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
        ScaleIntensityRanged(keys=["image"], a_min=np.float32(0), a_max=np.float32(255)),
        ToTensord(keys=["image"]),
    ])

    dataset = Dataset(data=datalist, transform=infer_transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False if args.workers == 0 else True,
        collate_fn=list_data_collate,
    )

    model = milmodel.MILModel(num_classes=args.num_classes, pretrained=True, mil_mode=args.mil_mode, trans_dropout=0.0)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise ValueError("--checkpoint is required for inference and must point to a valid file")

    device = torch.device("cuda", args.device) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    # Derive experiment name from checkpoint path
    experiment_name = re.sub(".pt","",os.path.dirname(args.checkpoint))

    out_csv = os.path.join(args.results_dir, f"{experiment_name}.csv")
    print(f"Writing predictions to: {out_csv}")

    rows = [("path", "label")]
    start_time = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            aggressive_memory_cleanup() if (idx % 10 == 0 and idx > 0) else None

            item_meta = dataset.data[idx]
            img_path = item_meta.get("image") if isinstance(item_meta, dict) else None
            if img_path is None:
                img_path = batch.get("image_meta_dict", [{}])[0].get("filename_or_obj", f"item_{idx}")

            data = batch["image"].as_subclass(torch.Tensor).to(device)

            if args.tile_count is not None and data.shape[1] > args.tile_count:
                perm = torch.randperm(data.shape[1])[: args.tile_count]
                data = data[:, perm]

            max_tiles = args.max_tiles if args.max_tiles is not None else args.tile_count

            with autocast("cuda", enabled=args.amp):
                if max_tiles is not None and data.shape[1] > max_tiles:
                    logits_chunks = []
                    for i in range(int(np.ceil(data.shape[1] / float(max_tiles)))):
                        sl = data[:, i * max_tiles : (i + 1) * max_tiles]
                        logits_chunks.append(model(sl, no_head=False).detach().cpu())
                        del sl
                        torch.cuda.empty_cache()
                    logits = torch.cat(logits_chunks, dim=1).to(device)
                else:
                    logits = model(data)

            pred_score = logits.sigmoid().sum(1).detach().cpu().numpy()
            pred_label = int(np.round(pred_score)[0])

            rows.append((str(img_path), pred_label))

            del data, logits
            torch.cuda.empty_cache()

            if (idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {idx + 1}/{len(loader)} in {elapsed:.1f}s")
                start_time = time.time()

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved {len(rows)-1} predictions to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Inference-only MIL WSI classifier → CSV export")
    p.add_argument("--data_root", default="/data/liver_image_all/", type=str)
    p.add_argument("--dataset_json", required=True, type=str)
    p.add_argument("--split_key", default="validation", type=str)
    p.add_argument("--num_classes", default=2, type=int)
    p.add_argument("--mil_mode", default="att_trans", type=str)
    p.add_argument("--tile_count", default=45, type=int)
    p.add_argument("--tile_size", default=224, type=int)
    p.add_argument("--max_tiles", default=None, type=int)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--results_dir", default="./result", type=str)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
