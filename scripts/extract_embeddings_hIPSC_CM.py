#!/usr/bin/env python3
"""
Embedding Extraction Script for hIPSC_CM dataset using MIMIC (Frozen Encoder)
=============================================================================
This script:
1. Loads the hIPSC_CM dataset (e.g. hIPSC_CM_ej_cds_transformed.txt).
2. Parses the sequence tokens:
   - Strips exon junction markers ('ej', 'Gej', etc.).
   - Converts the raw nucleotide tokens into clean RNA sequences (A, C, G, U).
   - Optionally extracts the codon-aware CDS binary mask ('0' for UTR, '1' for CDS).
   - Enforces MIMIC's maximum context length (10,000 nt).
3. Passes the sequences through the frozen MIMIC foundation model encoder.
4. Extracts the joint representation as described in the MIMIC paper (Section D.3):
   - Register tokens [5 x 1536 = 7680 dim] + Mean-pooled RNA track [1536 dim] = 9216 dim.
   - Also provides 1536-dim RNA-mean and 3072-dim (register-mean + RNA-mean) representations.
5. Saves all embeddings and metadata in an NPZ archive at:
   /beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/mimic/mimic_embeddings_hIPSC_CM.npz

Usage:
------
# Basic run on SLURM cluster:
python scripts/extract_embeddings_hIPSC_CM.py

# With CDS mask conditioning:
python scripts/extract_embeddings_hIPSC_CM.py --include_cds

# With custom local weights directory:
python scripts/extract_embeddings_hIPSC_CM.py --weights_dir /path/to/mimic_weights
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure local mimic package is on sys.path if run from within repo or scripts dir
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from mimic import load_pretrained
except ImportError as e:
    raise ImportError(
        f"Could not import 'mimic'. Please install the package (`pip install -e .`) "
        f"or ensure {REPO_ROOT / 'src'} is in PYTHONPATH. Error: {e}"
    )


# Default Paths on the cluster
DEFAULT_DATA_PATH = "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hIPSC_CM_ej_cds_transformed.txt"
DEFAULT_OUTPUT_DIR = "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/mimic"
DEFAULT_SPLITS_PATH = "/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv"
DEFAULT_OUTPUT_FILENAME = "mimic_embeddings_hIPSC_CM.npz"


def parse_saluki_sequence_for_mimic(
    raw_seq: str,
    max_length: int = 10000,
) -> Tuple[str, str, int]:
    """
    Parses comma-separated Saluki/Orthrus tokens into MIMIC-compatible inputs.

    - Nucleotide sequence: tok[0], uppercased, T -> U.
    - CDS mask: In Saluki format, uppercase letters mark the first nucleotide
      of each 3-base codon. If present, all 3 bases of each codon are marked '1',
      while non-coding bases (UTRs or non-coding transcripts) are '0'.

    Returns:
        (rna_seq, cds_mask_str, original_length)
    """
    tokens = [tok.strip() for tok in raw_seq.split(",") if tok.strip()]
    orig_len = len(tokens)
    if orig_len == 0:
        return "", "", 0

    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    L = len(tokens)

    # 1. Clean RNA sequence (first character of token, uppercase, T -> U)
    clean_nucleotides = [tok[0].upper().replace("T", "U") for tok in tokens]
    rna_seq = "".join(clean_nucleotides)

    # 2. Extract CDS mask ('0' or '1' per nucleotide position)
    cds_mask = ["0"] * L
    i = 0
    while i < L:
        # Uppercase letter marks codon start (first nucleotide of triplet)
        if tokens[i][0].isupper():
            for offset in range(3):
                if i + offset < L:
                    cds_mask[i + offset] = "1"
            i += 3
        else:
            i += 1

    cds_mask_str = "".join(cds_mask)
    return rna_seq, cds_mask_str, orig_len


def build_batches_by_length(
    samples: List[Dict],
    batch_size: int = 4,
    max_tokens_per_batch: int = 24000,
) -> List[List[Dict]]:
    """
    Sorts samples by sequence length to minimize padding waste during batching,
    and creates batches respecting both max batch_size and max_tokens_per_batch.
    """
    # Sort indices by sequence length
    sorted_samples = sorted(samples, key=lambda s: len(s["rna_seq"]))

    batches = []
    current_batch = []
    current_max_len = 0

    for s in sorted_samples:
        seq_len = len(s["rna_seq"])
        prospective_max_len = max(current_max_len, seq_len)
        prospective_tokens = (len(current_batch) + 1) * prospective_max_len

        if len(current_batch) >= batch_size or (current_batch and prospective_tokens > max_tokens_per_batch):
            batches.append(current_batch)
            current_batch = [s]
            current_max_len = seq_len
        else:
            current_batch.append(s)
            current_max_len = prospective_max_len

    if current_batch:
        batches.append(current_batch)

    return batches


def extract_mimic_embeddings(
    samples: List[Dict],
    model: torch.nn.Module,
    include_cds: bool = False,
    batch_size: int = 4,
    max_tokens_per_batch: int = 24000,
    desc: str = "Extracting MIMIC Embeddings",
) -> Dict[str, np.ndarray]:
    """
    Runs batched inference through MIMIC frozen encoder and extracts representations.

    Returns dict with arrays:
    - 'concat_9216': register tokens [5*1536] concatenated with RNA mean-pool [1536] (Standard mRNABench)
    - 'rna_mean_1536': RNA track mean pooling only [1536]
    - 'reg_mean_1536': Register tokens mean pooling [1536]
    - 'concat_3072': Register mean [1536] + RNA mean [1536] = [3072]
    """
    N = len(samples)
    emb_9216 = np.zeros((N, 9216), dtype=np.float32)
    emb_rna_mean = np.zeros((N, 1536), dtype=np.float32)
    emb_reg_mean = np.zeros((N, 1536), dtype=np.float32)
    emb_3072 = np.zeros((N, 3072), dtype=np.float32)

    batches = build_batches_by_length(
        samples, batch_size=batch_size, max_tokens_per_batch=max_tokens_per_batch
    )

    with torch.no_grad():
        for batch in tqdm(batches, desc=desc):
            # Prepare input batch dicts for MIMIC
            model_batch = []
            for item in batch:
                inp = {"rna_seq": item["rna_seq"]}
                if include_cds and len(item.get("cds", "")) == len(item["rna_seq"]):
                    inp["cds"] = item["cds"]
                model_batch.append(inp)

            # Stage inputs
            model.input(model_batch)

            # Forward pass through frozen encoder
            # return_full: 'full' tensor has shape [B, num_registers + max_L, D]
            # return_register: 'register' tensor has shape [B, num_registers, D]
            # return_modality: per-sample dict mapping group ('dna/rna') to unpadded tokens [L_i, D]
            reps = model.embed(return_full=True, return_register=True, return_modality=True)

            register_tensor = reps["register"]  # [B, 5, 1536]
            modality_dict = reps["modality"]    # {i: {'dna/rna': Tensor[L_i, 1536]}}

            for b_idx, item in enumerate(batch):
                orig_i = item["orig_idx"]

                # 1. Register representations
                reg_b = register_tensor[b_idx]  # [5, 1536]
                reg_flat = reg_b.flatten().cpu().numpy()  # [7680]
                reg_mean = reg_b.mean(dim=0).cpu().numpy()  # [1536]

                # 2. RNA track representation (unpadded mean across nucleotide positions)
                if "dna/rna" in modality_dict[b_idx]:
                    rna_tokens = modality_dict[b_idx]["dna/rna"]
                    rna_mean = rna_tokens.mean(dim=0).cpu().numpy()  # [1536]
                else:
                    # Fallback to zero if group was empty
                    rna_mean = np.zeros(1536, dtype=np.float32)

                # 3. Concatenation as used in mRNABench (Section D.3):
                # Concat register tokens and mean-pooled RNA track
                concat_vec = np.concatenate([reg_flat, rna_mean], axis=0)  # 7680 + 1536 = 9216
                concat_3072_vec = np.concatenate([reg_mean, rna_mean], axis=0)  # 1536 + 1536 = 3072

                emb_9216[orig_i] = concat_vec
                emb_rna_mean[orig_i] = rna_mean
                emb_reg_mean[orig_i] = reg_mean
                emb_3072[orig_i] = concat_3072_vec

            # Clear cache between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "concat_9216": emb_9216,
        "rna_mean_1536": emb_rna_mean,
        "reg_mean_1536": emb_reg_mean,
        "concat_3072": emb_3072,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC frozen encoder embeddings for hIPSC_CM dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to hIPSC_CM transformed TSV file (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save embeddings (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Filename of the NPZ archive (default: {DEFAULT_OUTPUT_FILENAME})",
    )
    parser.add_argument(
        "--splits_path",
        type=str,
        default=DEFAULT_SPLITS_PATH,
        help=f"Optional path to 10-fold split lookup CSV (default: {DEFAULT_SPLITS_PATH})",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="1.0",
        help="MIMIC model version to download/load (default: 1.0)",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="Optional local path to weights directory containing config.json and model.safetensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Max batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=24000,
        help="Max total tokens in a single batch to avoid VRAM overflow (default: 24000)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10000,
        help="Maximum transcript sequence length for MIMIC (default: 10000)",
    )
    parser.add_argument(
        "--include_cds",
        action="store_true",
        help="If set, also includes the binary CDS mask modality alongside rna_seq",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / args.output_filename
    if args.include_cds and "_cds" not in save_path.stem:
        save_path = output_dir / f"{save_path.stem}_cds{save_path.suffix}"

    print("=" * 75)
    print("        MIMIC Embedding Extraction Pipeline for hIPSC_CM        ")
    print("=" * 75)
    print(f"Data file:             {data_path}")
    print(f"Output file:           {save_path}")
    print(f"MIMIC Version:         {args.model_version}")
    print(f"Local Weights:         {args.weights_dir or 'Hugging Face Hub (polymathic-ai/MIMIC)'}")
    print(f"Device:                {args.device}")
    print(f"Conditioning:          {'rna_seq + cds' if args.include_cds else 'rna_seq only'}")
    print(f"Max sequence len:      {args.max_length} nt")
    print(f"Batch size:            {args.batch_size} (max {args.max_tokens_per_batch} tokens/batch)")

    # 1. Load Data
    print(f"\n[1/4] Loading dataset from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file '{data_path}' does not exist! Please check the path."
        )

    df = pd.read_csv(data_path, sep="\t")
    num_samples = len(df)
    print(f"Loaded {num_samples} transcripts. Columns: {list(df.columns)}")

    if "sequence" not in df.columns:
        raise KeyError(f"Column 'sequence' not found in dataset! Available: {list(df.columns)}")

    # 2. Parse sequences
    print(f"\n[2/4] Parsing and formatting sequences for MIMIC...")
    raw_seqs = df["sequence"].astype(str).values
    parsed_samples = []
    truncated_count = 0
    seq_lens = []

    for idx, raw in enumerate(tqdm(raw_seqs, desc="Parsing sequences")):
        rna_seq, cds_mask, orig_len = parse_saluki_sequence_for_mimic(
            raw, max_length=args.max_length
        )
        if orig_len > args.max_length:
            truncated_count += 1
        seq_lens.append(len(rna_seq))
        parsed_samples.append({
            "orig_idx": idx,
            "rna_seq": rna_seq,
            "cds": cds_mask,
        })

    seq_lens = np.array(seq_lens, dtype=np.int32)
    if truncated_count > 0:
        print(f"Notice: {truncated_count} sequences exceeded {args.max_length} nt and were truncated.")
    print(f"Sequence length stats: min={seq_lens.min()}, mean={seq_lens.mean():.1f}, max={seq_lens.max()}")

    # 3. Load MIMIC Model
    print(f"\n[3/4] Loading pretrained MIMIC model (Frozen Encoder)...")
    t0 = time.time()
    model = load_pretrained(
        version=args.model_version,
        local_path=args.weights_dir,
        device=args.device,
    )
    # Ensure encoder is frozen
    model.freeze_encoder(freeze_embeddings=True)
    model.eval()
    print(f"Model loaded successfully in {time.time() - t0:.2f}s on {model.device}.")

    # 4. Extract Embeddings
    print(f"\n[4/4] Extracting embeddings across {num_samples} samples...")
    t_start = time.time()
    embeddings_dict = extract_mimic_embeddings(
        samples=parsed_samples,
        model=model,
        include_cds=args.include_cds,
        batch_size=args.batch_size,
        max_tokens_per_batch=args.max_tokens_per_batch,
        desc="Extracting MIMIC Embeddings",
    )
    elapsed = time.time() - t_start
    print(f"Inference completed in {elapsed:.2f}s ({elapsed / num_samples:.4f}s / sample).")

    # Match splits if lookup exists
    splits_path = Path(args.splits_path)
    sample_splits = np.full(num_samples, -1, dtype=np.int32)
    if splits_path.exists():
        try:
            lookup_df = pd.read_csv(splits_path)
            if "ensembl_transcript_id" in lookup_df.columns and "split" in lookup_df.columns:
                tx_to_split = dict(
                    zip(lookup_df["ensembl_transcript_id"].astype(str).str.strip(),
                        lookup_df["split"].astype(int))
                )
                tx_ids = df["ensembl_transcript_id"].astype(str).str.strip().values
                sample_splits = np.array([tx_to_split.get(t, -1) for t in tx_ids], dtype=np.int32)
                matched_splits = (sample_splits >= 0).sum()
                print(f"Matched {matched_splits}/{num_samples} transcripts with 10-fold lookup table.")
        except Exception as e:
            print(f"[Warning] Could not process split lookup table: {e}")

    # Extract target metadata
    half_lives = (
        df["half_life"].astype(np.float32).values
        if "half_life" in df.columns
        else np.full(num_samples, np.nan, dtype=np.float32)
    )
    half_lives_transformed = (
        df["half_life_transformed"].astype(np.float32).values
        if "half_life_transformed" in df.columns
        else np.full(num_samples, np.nan, dtype=np.float32)
    )
    rates = (
        df["rate"].astype(np.float32).values
        if "rate" in df.columns
        else np.full(num_samples, np.nan, dtype=np.float32)
    )
    transcript_ids = (
        df["ensembl_transcript_id"].astype(str).values
        if "ensembl_transcript_id" in df.columns
        else np.array([""] * num_samples)
    )
    gene_ids = (
        df["ensembl_gene_id"].astype(str).values
        if "ensembl_gene_id" in df.columns
        else np.array([""] * num_samples)
    )
    gene_symbols = (
        df["hgnc_symbol"].astype(str).values
        if "hgnc_symbol" in df.columns
        else np.array([""] * num_samples)
    )
    biotypes = (
        df["transcript_biotype"].astype(str).values
        if "transcript_biotype" in df.columns
        else np.array([""] * num_samples)
    )

    # Save to NPZ
    print(f"\nSaving NPZ archive to: {save_path} ...")
    np.savez_compressed(
        save_path,
        embeddings=embeddings_dict["concat_9216"],  # Primary: register [7680] + rna_mean [1536] = 9216
        embeddings_rna_mean=embeddings_dict["rna_mean_1536"],  # 1536
        embeddings_reg_mean=embeddings_dict["reg_mean_1536"],  # 1536
        embeddings_3072=embeddings_dict["concat_3072"],        # 3072
        half_life=half_lives,
        half_life_transformed=half_lives_transformed,
        rate=rates,
        ensembl_transcript_id=transcript_ids,
        ensembl_gene_id=gene_ids,
        hgnc_symbol=gene_symbols,
        transcript_biotype=biotypes,
        seq_lens=seq_lens,
        split=sample_splits,
        model_version=str(args.model_version),
        include_cds=args.include_cds,
        max_length=args.max_length,
    )

    print(f"\n[✓] Done! Embeddings successfully saved to:")
    print(f"    Path:             {save_path}")
    print(f"    Primary Shape:    {embeddings_dict['concat_9216'].shape} (9216 dimensions)")
    print(f"    Alternative Shapes: 1536 (RNA mean), 3072 (Reg mean + RNA mean)")


if __name__ == "__main__":
    main()
