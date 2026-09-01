import argparse
import datetime
import logging
from pathlib import Path
import os
from tqdm import tqdm
import glob
import torch
import numpy as np
import awkward as ak
from typing import Union

from embedding.utils.cfg_handler import data_config
from embedding.utils.data_utils import softkill

# Making it possible to log to file as well as to console
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("converterHLT")

def gather_pfcands_collide1m(path: str, max_events: int = -1) -> ak.Array:
    """Read FullReco PF candidates from a collide1m parquet file.

    The parquet files store PF candidates in a single nested struct column
    ``FullReco_PFPart`` whose sub-fields are per-event jagged lists:
    ``pt, eta, phi, dxy, dxysig, pdgId, charge, ...``.

    Returns an ak.Array with fields (pt, eta, phi, dxy, dxysig, is_pf, pdgId)
    matching PFPreProcessor's expected input order.
    """
    arr = ak.from_parquet(path, columns=["FullReco_PFPart"])
    if max_events > 0:
        arr = arr[:max_events]

    part = arr["FullReco_PFPart"]

    def f32(x):
        return ak.values_astype(x, np.float32)

    pt     = f32(part["pt"])
    eta    = f32(part["eta"])
    phi    = f32(part["phi"])
    dxy    = f32(part["dxy"])
    dxysig = f32(part["dxysig"])
    # pdgId is stored unsigned (abs value); re-apply the sign from the charge
    # field so PFPreProcessor can derive particle charge from pdgId sign.
    abs_pid = ak.values_astype(part["pdgId"], np.int64)
    charge  = ak.values_astype(part["charge"], np.int64)
    signed_pid = ak.where(charge != 0, abs_pid * charge, abs_pid)

    return ak.zip({
        "pt":     pt,
        "eta":    eta,
        "phi":    phi,
        "dxy":    dxy,
        "dxysig": dxysig,
        "is_pf":  ak.ones_like(pt),
        "pdgId":  f32(signed_pid),
    })

def process_pfcands(
        combined: ak.Array,
        label: int,
        n_objects: int = 200,
        sk_cell_size: Union[float, None] = None,
        sort_by_pt: bool = False,
    ) -> torch.Tensor:

    if sort_by_pt:
        logger.info("Sorting PF candidates by pt")
        combined = combined[ak.argsort(combined.pt, axis=1, ascending=False)]

    counts = ak.num(combined.pt, axis=1)
    clipped_events = counts > n_objects
    if ak.any(clipped_events):
        lost = int(ak.sum(counts[clipped_events] - n_objects))
        logger.warning(f"{int(ak.sum(clipped_events))} events clipped; {lost} particles lost. Increase n_objects to avoid this.")
    padded = ak.pad_none(combined, n_objects, axis=1, clip=True)
    array = np.stack(
        [ak.to_numpy(padded[field]) for field in padded.fields],
        axis=-1,
    )
    tensor = torch.tensor(array, dtype=torch.float32)
    label_tensor = torch.full((tensor.shape[0], n_objects, 1), label, dtype=torch.float32)

    if sk_cell_size is not None:
        tensor = softkill(tensor, cell_size=sk_cell_size)

    return torch.cat([tensor, label_tensor], dim=-1)

def main(cfg: data_config, overwrite: bool = False):
    """Convert collide1m parquet files to fixed-shape PyTorch tensors and save train/test splits."""

    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    sample_dir = Path(cfg["sample_dir"]).expanduser()
    n_objects = cfg.get("n_objects", 500)
    nevents_per_class = cfg.get("nevents_per_class", -1)
    sk_cell_size = cfg.get("sk_spacing", None)
    sort_by_pt = cfg.get("sort_by_pt", True)
    store_by_class = cfg.get("store_by_class", False)
    split = cfg.get("split", None)
    logger.info(f"Soft-kill cell size: {sk_cell_size}")

    if split and store_by_class:
        raise ValueError("Cannot use both split and store_by_class options at the same time.")

    tensors = {}
    file_label_tuples = cfg.get_file_label_map()
    for entry in tqdm(file_label_tuples, desc="Processing files"):
        file_name, label = entry

        folder_path = sample_dir / file_name
        file_paths = sorted(Path(p) for p in glob.glob(str(folder_path / "*.parquet")))
        if not file_paths:
            logger.warning(f"No parquet files found in {folder_path}")
            continue

        n_events_left = nevents_per_class
        for path in file_paths:
            combined = gather_pfcands_collide1m(str(path), max_events=n_events_left)
            event_tensor = process_pfcands(
                combined,
                label=label,
                n_objects=n_objects,
                sk_cell_size=sk_cell_size,
                sort_by_pt=sort_by_pt,
            )

            tensors[label] = tensors.get(label, []) + [event_tensor]
            n_events_left -= event_tensor.shape[0]
            if nevents_per_class > 0 and n_events_left <= 0:
                break

    class_tensors = {label: torch.cat(chunks, dim=0) for label, chunks in tensors.items()}

    for label in class_tensors:
        # Shuffle across the per-sample chunks before capping so a class made of
        # several samples doesn't get filled entirely from whichever sample comes
        # first (and is largest). With nevents_per_class <= 0 nothing is dropped.
        class_tensors[label] = class_tensors[label][torch.randperm(class_tensors[label].shape[0])]
        if nevents_per_class > 0:
            class_tensors[label] = class_tensors[label][:nevents_per_class]

    total_num_events = sum(tensor.shape[0] for tensor in class_tensors.values())
    logger.info("Class event counts:")
    for label in class_tensors:
        logger.info(f"  Label {label}: {class_tensors[label].shape[0]} events ({round(class_tensors[label].shape[0] / total_num_events * 100, 2)}%)")

    full_tensor = torch.cat([class_tensors[label] for label in class_tensors], dim=0)
    full_tensor = torch.nan_to_num(full_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    full_tensor = full_tensor[torch.randperm(full_tensor.shape[0])] # Randomize order

    output_prefix = cfg.get_ds_name()
    if output_prefix == "":
        output_prefix = "embedding_hlt_ssl"
        logger.warning(f"Dataset name not found in config; using default {output_prefix}.")

    out_path = Path(cfg.get("out_path", "./")).expanduser()
    os.makedirs(out_path, exist_ok=True)

    if split is not None:
        # One file per train/test split, each w/ mixed classes
        split_idx = int(cfg.get("split", 0.8) * full_tensor.shape[0])
        train_fname = out_path / f"{output_prefix}_train.pt"
        test_fname = out_path / f"{output_prefix}_test.pt"
        file_exists = train_fname.exists() or test_fname.exists()
        if file_exists and not overwrite:
            raise FileExistsError(f"Output files with prefix {output_prefix} already exist in {os.fspath(out_path)}. Use --overwrite to overwrite.")
        train_tensor = full_tensor[:split_idx]
        test_tensor = full_tensor[split_idx:]

        n_events_per_class_train = torch.bincount(train_tensor[:, 0, -1].to(torch.int64))
        n_events_per_class_test = torch.bincount(test_tensor[:, 0, -1].to(torch.int64))
        logger.info("Train class event counts:")
        for i in range(len(n_events_per_class_train)):
            logger.info(f"  Label {i}: {n_events_per_class_train[i]} events ({round(n_events_per_class_train[i].item() / train_tensor.shape[0] * 100, 2)}%)")
        logger.info("Test class event counts:")
        for i in range(len(n_events_per_class_test)):
            logger.info(f"  Label {i}: {n_events_per_class_test[i]} events ({round(n_events_per_class_test[i].item() / test_tensor.shape[0] * 100, 2)}%)")

        torch.save(train_tensor, os.fspath(train_fname))
        torch.save(test_tensor, os.fspath(test_fname))
    elif store_by_class:
        # One file per class
        label_name_map = cfg.get_label_name_map()
        for label, name in label_name_map.items():
            full_fname = out_path / f"{output_prefix}_{name}_testds.pt"
            if not overwrite and full_fname.exists():
                raise FileExistsError(f"Output file with prefix {output_prefix}_{name}_testds.pt already exists in {os.fspath(out_path)}. Use --overwrite to overwrite.")
            event_labels = full_tensor[:, 0, -1].to(torch.int64)
            class_tensor = full_tensor[event_labels == int(label)]
            torch.save(class_tensor, os.fspath(full_fname))
    else:
        # All classes mixed in one file
        full_fname = out_path / f"{output_prefix}.pt"
        if not overwrite and full_fname.exists():
            raise FileExistsError(f"Output file with prefix {output_prefix}.pt already exists in {os.fspath(out_path)}. Use --overwrite to overwrite.")
        torch.save(full_tensor, os.fspath(full_fname))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the config .yaml file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    cfg = data_config(args.config)

    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/converterHLT_{cfg.get_ds_name()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting conversion with config: {args.config}")

    main(cfg, overwrite=args.overwrite)
