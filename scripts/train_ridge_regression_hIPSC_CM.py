#!/usr/bin/env python3
"""
Ridge regression training & evaluation on extracted MIMIC embeddings for the hIPSC_CM dataset.
Supports the 3 MIMIC multimodal embedding representations:
- 'protein':     Uses reconstructed protein sequence embeddings (9216-dim)
- 'rna':         Uses RNA nucleotide sequence embeddings (9216-dim)
- 'rna_protein': Joint multimodal conditioning embeddings (10752-dim)

Integrates standardized 10-fold gene-grouped split lookup table:
- Evaluates on test splits [8, 9]
- Conducts custom 4-fold Cross-Validation over splits [0..7]:
    Fold 0: Train [0, 1, 2, 3, 4, 5], Val [6, 7]
    Fold 1: Train [2, 3, 4, 5, 6, 7], Val [0, 1]
    Fold 2: Train [0, 1, 4, 5, 6, 7], Val [2, 3]
    Fold 3: Train [0, 1, 2, 3, 6, 7], Val [4, 5]
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Headless backend for cluster servers without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Default Paths on the cluster
DEFAULT_EMBEDDING_PATHS = {
    "protein": Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/mimic/mimic_embeddings_hIPSC_CM_protein.npz"),
    "rna": Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/mimic/mimic_embeddings_hIPSC_CM_rna.npz"),
    "rna_protein": Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/mimic/mimic_embeddings_hIPSC_CM_rna_protein.npz"),
}
DEFAULT_LOOKUP_PATH = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/hipsc_cm_10folds_lookup.csv")
DEFAULT_OUTPUT_DIR = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/results/MIMIC/hIPSC_CM")
DEFAULT_TRANSFORMATION_PARAMS_PATH = Path("/beegfs/prj/RNA_NLP/FlorianMasterThesis/code/data/hIPSC_CM/transformation_params.json")

# Alpha grid according to linear_probe_eval.py / Orthrus evaluation protocol
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Calculates regression metrics including correlations."""
    p_corr, p_val = pearsonr(y_true, y_pred)
    s_corr, s_val = spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    prefix_str = f"{prefix}_" if prefix else ""
    return {
        f"{prefix_str}pearson_r": float(p_corr),
        f"{prefix_str}pearson_pvalue": float(p_val),
        f"{prefix_str}spearman_rho": float(s_corr),
        f"{prefix_str}spearman_pvalue": float(s_val),
        f"{prefix_str}mse": float(mse),
        f"{prefix_str}rmse": rmse,
        f"{prefix_str}mae": float(mae),
        f"{prefix_str}r2": float(r2),
    }


def print_metrics(metrics: dict, title: str):
    """Formatted console output for metrics."""
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if "pvalue" in k:
            print(f"  {k:26s}: {v:.3e}")
        else:
            print(f"  {k:26s}: {v:.4f}")


def load_transformation_params(params_path: Path):
    """
    Loads mu_log, sigma_log, and pseudocount from transformation_params.json if available.
    Returns (mu_log, sigma_log, pseudocount) or None if file does not exist.
    """
    if not params_path.exists():
        return None

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        if "mu_log" in params and "sigma_log" in params:
            mu_log = float(params["mu_log"])
            sigma_log = float(params["sigma_log"])
            pseudocount = float(params.get("pseudocount", 0.1))
            return mu_log, sigma_log, pseudocount, params_path
    except Exception as e:
        print(f"[Notice] Could not parse transformation parameters: {e}")
    return None


def inverse_transform_half_life(
    y_transformed: np.ndarray,
    mu: float,
    sigma: float,
    pseudocount: float = 0.1,
) -> np.ndarray:
    """
    Inverts the log + z-score transformation:
      y_log = (y_transformed * sigma) + mu
      y_raw = exp(y_log) - pseudocount
    Clips output to >= 0 to avoid negative physical half-lives.
    """
    y_log = (y_transformed * sigma) + mu
    y_raw = np.exp(y_log) - pseudocount
    return np.clip(y_raw, a_min=0.0, a_max=None)


def load_mimic_npz(file_path: Path, target_col: str) -> dict:
    """Loads a MIMIC NPZ file containing embeddings and metadata."""
    if not file_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {file_path}")

    data = np.load(file_path, allow_pickle=True)
    available_keys = list(data.keys())

    if "embeddings" not in data:
        raise KeyError(f"'embeddings' not found in NPZ. Available keys: {available_keys}")

    if target_col not in data:
        raise KeyError(
            f"Target variable '{target_col}' not found in NPZ archive. "
            f"Available keys: {available_keys}"
        )

    targets = data[target_col].astype(np.float32)

    # Gene column for group split (prefer hgnc_symbol, else ensembl_gene_id, else gene)
    if "hgnc_symbol" in data:
        genes = data["hgnc_symbol"].astype(str)
    elif "ensembl_gene_id" in data:
        genes = data["ensembl_gene_id"].astype(str)
    elif "genes" in data:
        genes = data["genes"].astype(str)
    else:
        genes = np.array([f"gene_{i}" for i in range(len(targets))])

    # Transcript IDs
    if "ensembl_transcript_id" in data:
        transcript_ids = data["ensembl_transcript_id"].astype(str)
    elif "transcript_id" in data:
        transcript_ids = data["transcript_id"].astype(str)
    elif "tx_ids" in data:
        transcript_ids = data["tx_ids"].astype(str)
    else:
        transcript_ids = np.array([f"tx_{i}" for i in range(len(targets))])

    raw_half_life = None
    if "half_life" in data:
        raw_half_life = data["half_life"].astype(np.float32)

    has_cds = None
    if "has_cds" in data:
        has_cds = data["has_cds"].astype(bool)

    mode = str(data["mode"]) if "mode" in data else None

    return {
        "embeddings": data["embeddings"],
        "targets": targets,
        "genes": genes,
        "transcript_ids": transcript_ids,
        "raw_half_life": raw_half_life,
        "has_cds": has_cds,
        "mode": mode,
        "archive_keys": available_keys,
    }


def train_and_evaluate_mode(
    mode_name: str,
    emb_path: Path,
    lookup_path: Path,
    out_dir: Path,
    target_col: str = "half_life_transformed",
    random_state: int = 42,
    plot: bool = False,
) -> dict:
    """Trains and evaluates Ridge regression for a single MIMIC embedding mode."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"   MIMIC Ridge Regression: {mode_name.upper()} (hIPSC_CM)")
    print("=" * 75)
    print(f"Embeddings file:    {emb_path}")
    print(f"Target variable:    {target_col}")
    print(f"Lookup splits file: {lookup_path}")
    print(f"Output directory:   {out_dir}")
    print(f"Random state:       {random_state}")

    data = load_mimic_npz(emb_path, target_col=target_col)
    X = data["embeddings"]
    y = data["targets"]
    genes = data["genes"]
    transcript_ids = data["transcript_ids"]
    raw_half_life = data.get("raw_half_life")
    has_cds = data.get("has_cds")

    # 1. Filter invalid target values (NaN / Inf) and unencoded / all-zero embeddings
    valid_target = ~np.isnan(y) & ~np.isinf(y)
    valid_emb = np.any(X != 0, axis=1)
    valid_mask = valid_target & valid_emb

    if not np.all(valid_mask):
        num_invalid = int(np.sum(~valid_mask))
        num_inv_targets = int(np.sum(~valid_target))
        num_zero_embs = int(np.sum(~valid_emb))
        print(f"[Data Cleaning] Filtered out {num_invalid} samples ({num_inv_targets} invalid targets, {num_zero_embs} zero-norm embeddings).")
        X = X[valid_mask]
        y = y[valid_mask]
        genes = genes[valid_mask]
        transcript_ids = transcript_ids[valid_mask]
        if raw_half_life is not None:
            raw_half_life = raw_half_life[valid_mask]

    print(f"Valid samples:       {len(y)}")
    print(f"Embedding dimension: {X.shape[1]}")
    print(f"Unique genes:        {len(np.unique(genes))}")

    if len(y) == 0:
        raise ValueError(f"No valid data points found for target '{target_col}' in {emb_path}.")

    # 2. Match splits from standardized 10-Fold Lookup Table
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"[Error] Standardized splits lookup table not found at: {lookup_path}! "
            f"A valid lookup table is strictly required."
        )

    lookup_df = pd.read_csv(lookup_path)
    lookup_tx_col = "ensembl_transcript_id" if "ensembl_transcript_id" in lookup_df.columns else "transcript_id"
    if lookup_tx_col not in lookup_df.columns:
        raise KeyError(
            f"[Error] Transcript ID column ('ensembl_transcript_id' or 'transcript_id') not found in lookup table {lookup_path}. "
            f"Columns: {list(lookup_df.columns)}"
        )
    if "split" not in lookup_df.columns:
        raise KeyError(
            f"[Error] 'split' column not found in lookup table {lookup_path}. Columns: {list(lookup_df.columns)}"
        )

    tx_to_split = dict(zip(lookup_df[lookup_tx_col].astype(str).str.strip(), lookup_df["split"].astype(int)))
    sample_splits = np.array([tx_to_split.get(str(t).strip(), -1) for t in transcript_ids])

    unmatched_count = int(np.sum(sample_splits == -1))
    if unmatched_count > 0:
        print(f"[Warning] {unmatched_count} transcripts not found in lookup table! Filtering them out.")
        matched_mask = sample_splits != -1
        X = X[matched_mask]
        y = y[matched_mask]
        genes = genes[matched_mask]
        transcript_ids = transcript_ids[matched_mask]
        sample_splits = sample_splits[matched_mask]
        if raw_half_life is not None:
            raw_half_life = raw_half_life[matched_mask]

    # 3. Setup CV Pool (Splits 0-7) and Holdout Test Set (Splits 8 & 9)
    test_mask = np.isin(sample_splits, [8, 9])
    cv_mask = np.isin(sample_splits, [0, 1, 2, 3, 4, 5, 6, 7])

    X_train_cv, y_train_cv = X[cv_mask], y[cv_mask]
    cv_splits = sample_splits[cv_mask]
    train_genes = genes[cv_mask]

    X_test, y_test = X[test_mask], y[test_mask]
    test_genes = genes[test_mask]
    test_tx = transcript_ids[test_mask]
    y_test_raw_true = raw_half_life[test_mask] if raw_half_life is not None else None

    print(f"\n[Split Breakdown]")
    print(f"  CV Pool (Folds 0-7): {len(y_train_cv)} samples ({len(np.unique(train_genes))} unique genes)")
    print(f"  Test Set (Folds 8-9): {len(y_test)} samples ({len(np.unique(test_genes))} unique genes)")

    # 4. Standard 4-Fold Cross-Validation over splits [0..7]
    cv_fold_definitions = [
        {"name": "Fold 0", "train_splits": [0, 1, 2, 3, 4, 5], "val_splits": [6, 7]},
        {"name": "Fold 1", "train_splits": [2, 3, 4, 5, 6, 7], "val_splits": [0, 1]},
        {"name": "Fold 2", "train_splits": [0, 1, 4, 5, 6, 7], "val_splits": [2, 3]},
        {"name": "Fold 3", "train_splits": [0, 1, 2, 3, 6, 7], "val_splits": [4, 5]},
    ]

    custom_cv = []
    for fold_def in cv_fold_definitions:
        tr_idx = np.where(np.isin(cv_splits, fold_def["train_splits"]))[0]
        val_idx = np.where(np.isin(cv_splits, fold_def["val_splits"]))[0]
        custom_cv.append((tr_idx, val_idx))

    print(f"\nTraining RidgeCV across custom 4-Fold Cross-Validation (Alphas: {DEFAULT_ALPHAS})...")
    model = RidgeCV(alphas=DEFAULT_ALPHAS, cv=custom_cv)
    model.fit(X_train_cv, y_train_cv)

    best_alpha = float(model.alpha_)
    print(f"\n>>> Selected Optimal Alpha: {best_alpha:.4e} <<<")

    # Evaluate individual CV folds with optimal alpha
    print("\n" + "-" * 65)
    print("          4-FOLD CROSS-VALIDATION DIAGNOSTICS (VAL FOLDS)         ")
    print("-" * 65)
    print(f"{'Fold':<10} | {'Train Folds':<18} | {'Val Folds':<12} | {'Val Pearson r':<14} | {'Val RMSE':<10}")
    print("-" * 65)

    fold_evaluations = []
    val_pearsons = []
    val_rmses = []
    val_spearmans = []

    for f_idx, fold_def in enumerate(cv_fold_definitions):
        tr_idx, val_idx = custom_cv[f_idx]
        fold_ridge = Ridge(alpha=best_alpha, random_state=random_state)
        fold_ridge.fit(X_train_cv[tr_idx], y_train_cv[tr_idx])
        val_pred = fold_ridge.predict(X_train_cv[val_idx])

        f_metrics = calculate_metrics(y_train_cv[val_idx], val_pred, prefix=f"fold_{f_idx}")
        fold_evaluations.append({
            "fold_name": fold_def["name"],
            "train_splits": fold_def["train_splits"],
            "val_splits": fold_def["val_splits"],
            "val_samples": int(len(val_idx)),
            **f_metrics,
        })
        r_val = f_metrics[f"fold_{f_idx}_pearson_r"]
        rmse_val = f_metrics[f"fold_{f_idx}_rmse"]
        s_val = f_metrics[f"fold_{f_idx}_spearman_rho"]
        val_pearsons.append(r_val)
        val_rmses.append(rmse_val)
        val_spearmans.append(s_val)

        tr_str = ",".join(map(str, fold_def["train_splits"]))
        val_str = ",".join(map(str, fold_def["val_splits"]))
        print(f"{fold_def['name']:<10} | {tr_str:<18} | {val_str:<12} | {r_val:>12.4f}  | {rmse_val:>8.4f}")

    print("-" * 65)
    mean_cv_r = float(np.mean(val_pearsons))
    std_cv_r = float(np.std(val_pearsons))
    mean_cv_rmse = float(np.mean(val_rmses))
    std_cv_rmse = float(np.std(val_rmses))
    print(f"{'Mean ± Std':<10} | {'-':<18} | {'-':<12} | {mean_cv_r:>7.4f} ± {std_cv_r:.4f} | {mean_cv_rmse:>6.4f} ± {std_cv_rmse:.4f}")
    print("-" * 65)

    # Train predictions on entire CV pool (0-7)
    y_train_pred = model.predict(X_train_cv)
    train_metrics = calculate_metrics(y_train_cv, y_train_pred, prefix="train_cv_pool")

    # 5. Evaluate on Holdout Test Set (Splits 8 & 9)
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred, prefix="test")

    # Optional inverse transformation for half_life_transformed
    test_raw_metrics = {}
    y_test_pred_raw = None
    trans_info = None

    if target_col == "half_life_transformed":
        loaded_params = load_transformation_params(DEFAULT_TRANSFORMATION_PARAMS_PATH)
        if loaded_params is not None:
            mu_log, sigma_log, pseudocount, resolved_params_path = loaded_params
            print(f"\n[Transformation Params Loaded from: {resolved_params_path}]")
            print(f"  mu_log:      {mu_log:.6f} | sigma_log: {sigma_log:.6f} | pseudocount: {pseudocount}")

            y_test_pred_raw = inverse_transform_half_life(
                y_test_pred, mu=mu_log, sigma=sigma_log, pseudocount=pseudocount
            )
            if y_test_raw_true is None:
                y_test_raw_true = inverse_transform_half_life(
                    y_test, mu=mu_log, sigma=sigma_log, pseudocount=pseudocount
                )

            test_raw_metrics = calculate_metrics(y_test_raw_true, y_test_pred_raw, prefix="test_raw_hwz")
            trans_info = {
                "params_path": str(resolved_params_path),
                "mu_log": mu_log,
                "sigma_log": sigma_log,
                "pseudocount": pseudocount,
            }

    print_metrics(train_metrics, f"Training/CV Pool Metrics (Splits 0-7) ({target_col})")
    print_metrics(test_metrics, f"Test Set Metrics (Folds 8 & 9) - Transformed ({target_col})")
    if test_raw_metrics:
        print_metrics(test_raw_metrics, "Test Set Metrics (Folds 8 & 9) - Back-transformed (Hours)")

    # 6. Save Artifacts
    # Model
    model_file = out_dir / f"ridge_model_hIPSC_CM_{target_col}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to: {model_file}")

    # Predictions CSV
    pred_dict = {
        "transcript_id": test_tx,
        "gene": test_genes,
        "true_target": y_test,
        "predicted_target": y_test_pred,
        "residual": y_test - y_test_pred,
    }
    if y_test_pred_raw is not None and y_test_raw_true is not None:
        pred_dict["true_half_life_hours"] = y_test_raw_true
        pred_dict["predicted_half_life_hours"] = y_test_pred_raw
        pred_dict["residual_hours"] = y_test_raw_true - y_test_pred_raw

    pred_df = pd.DataFrame(pred_dict)
    pred_file = out_dir / f"predictions_hIPSC_CM_{target_col}.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Predictions saved to: {pred_file}")

    # Comprehensive Metrics JSON
    all_metrics = {
        "dataset": "hIPSC_CM",
        "model": "MIMIC",
        "mode": mode_name,
        "target_col": target_col,
        "embeddings_path": str(emb_path),
        "split_mechanism": "lookup_10folds",
        "evaluation_scheme": "4fold_cv_pool_0_7",
        "splits_lookup_path": str(lookup_path),
        "best_alpha": best_alpha,
        "cv_pool_size": int(len(y_train_cv)),
        "test_size": int(len(y_test)),
        "cv_pool_unique_genes": int(len(np.unique(train_genes))),
        "test_unique_genes": int(len(np.unique(test_genes))),
        "transformation_params": trans_info,
        "cv_folds_metrics": fold_evaluations,
        "cv_mean_pearson_r": mean_cv_r,
        "cv_std_pearson_r": std_cv_r,
        "cv_mean_rmse": mean_cv_rmse,
        "cv_std_rmse": std_cv_rmse,
        **train_metrics,
        **test_metrics,
        **test_raw_metrics,
    }
    metrics_file = out_dir / f"metrics_hIPSC_CM_{target_col}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_file}")

    # 7. Scatter Plot (y_true vs y_pred)
    if plot:
        try:
            emb_label = f"MIMIC ({mode_name.upper()})"
            if y_test_pred_raw is not None and y_test_raw_true is not None:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Subplot 1: Transformed Z-score
                ax1.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                r2 = test_metrics["test_r2"]
                rmse = test_metrics["test_rmse"]
                ax1.set_title(
                    f"Transformed Target (Z-Score)\n"
                    f"Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | RMSE = {rmse:.3f} | R² = {r2:.3f}",
                    fontsize=10,
                    fontweight="bold",
                )
                ax1.set_xlabel(f"True Value ({target_col})", fontsize=10)
                ax1.set_ylabel(f"Predicted Value ({target_col})", fontsize=10)
                min_v1 = min(float(np.min(y_test)), float(np.min(y_test_pred)))
                max_v1 = max(float(np.max(y_test)), float(np.max(y_test_pred)))
                margin1 = (max_v1 - min_v1) * 0.05
                ax1.plot([min_v1 - margin1, max_v1 + margin1], [min_v1 - margin1, max_v1 + margin1], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax1.legend()
                ax1.grid(True, linestyle="--", alpha=0.5)

                # Subplot 2: Back-transformed (Hours)
                ax2.scatter(y_test_raw_true, y_test_pred_raw, alpha=0.35, s=18, color="#2ca02c", edgecolors="none")
                p_r_raw = test_raw_metrics.get("test_raw_hwz_pearson_r", np.nan)
                s_rho_raw = test_raw_metrics.get("test_raw_hwz_spearman_rho", np.nan)
                rmse_raw = test_raw_metrics.get("test_raw_hwz_rmse", np.nan)
                mae_raw = test_raw_metrics.get("test_raw_hwz_mae", np.nan)
                ax2.set_title(
                    f"Back-transformed: Actual Half-Life (Hours)\n"
                    f"Pearson r = {p_r_raw:.3f} | Spearman rho = {s_rho_raw:.3f} | RMSE = {rmse_raw:.2f}h | MAE = {mae_raw:.2f}h",
                    fontsize=10,
                    fontweight="bold",
                )
                ax2.set_xlabel("True Half-Life (Hours)", fontsize=10)
                ax2.set_ylabel("Predicted Half-Life (Hours)", fontsize=10)
                min_v2 = min(float(np.min(y_test_raw_true)), float(np.min(y_test_pred_raw)))
                max_v2 = max(float(np.max(y_test_raw_true)), float(np.max(y_test_pred_raw)))
                margin2 = (max_v2 - min_v2) * 0.05
                ax2.plot([min_v2 - margin2, max_v2 + margin2], [min_v2 - margin2, max_v2 + margin2], "r--", linewidth=1.5, label="Ideal (y=x)")
                ax2.legend()
                ax2.grid(True, linestyle="--", alpha=0.5)

                fig.suptitle(f"{emb_label} -> hIPSC_CM Test Set Evaluation", fontsize=12, fontweight="bold")
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.scatter(y_test, y_test_pred, alpha=0.35, s=18, color="#1f77b4", edgecolors="none")
                p_r = test_metrics["test_pearson_r"]
                s_rho = test_metrics["test_spearman_rho"]
                r2 = test_metrics["test_r2"]

                ax.set_title(
                    f"{emb_label} -> hIPSC_CM ({target_col})\n"
                    f"Test Pearson r = {p_r:.3f} | Spearman rho = {s_rho:.3f} | R² = {r2:.3f}",
                    fontsize=11,
                    fontweight="bold",
                )
                ax.set_xlabel(f"True Value ({target_col})", fontsize=10)
                ax.set_ylabel(f"Predicted Value ({target_col})", fontsize=10)

                min_val = min(float(np.min(y_test)), float(np.min(y_test_pred)))
                max_val = max(float(np.max(y_test)), float(np.max(y_test_pred)))
                margin = (max_val - min_val) * 0.05
                ax.plot(
                    [min_val - margin, max_val + margin],
                    [min_val - margin, max_val + margin],
                    "r--",
                    linewidth=1.5,
                    label="Ideal (y=x)",
                )
                ax.legend()
                ax.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()

            plot_file = out_dir / f"scatter_hIPSC_CM_{target_col}.png"
            fig.savefig(plot_file, dpi=300)
            plt.close(fig)
            print(f"Scatter plot saved to: {plot_file}")
        except Exception as e:
            print(f"[Notice] Plot could not be created: {e}")

    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Ridge regression head on MIMIC hIPSC_CM embeddings"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "rna", "protein", "rna_protein"],
        default="all",
        help="MIMIC embedding mode to evaluate ('all', 'rna', 'protein', or 'rna_protein'). Default: 'all'",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help="Optional explicit path to an NPZ embedding file. If provided, overrides --mode.",
    )
    parser.add_argument(
        "--splits_lookup_path",
        type=str,
        default=str(DEFAULT_LOOKUP_PATH),
        help=f"Path to standardized 10-fold split lookup table CSV (default: {DEFAULT_LOOKUP_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="half_life_transformed",
        choices=["half_life_transformed", "half_life", "rate"],
        help="Target variable to train on (default: half_life_transformed)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optional: create scatter plot (y_true vs. y_pred) as PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_out_dir = Path(args.output_dir)
    lookup_path = Path(args.splits_lookup_path)

    # Determine tasks to run
    tasks = []
    if args.embeddings_path is not None:
        emb_path = Path(args.embeddings_path)
        # Try to infer mode from filename or data
        path_name = emb_path.stem.lower()
        if "rna_protein" in path_name:
            mode_name = "rna_protein"
        elif "protein" in path_name:
            mode_name = "protein"
        elif "rna" in path_name:
            mode_name = "rna"
        else:
            mode_name = path_name
        tasks.append((mode_name, emb_path, base_out_dir / mode_name))
    elif args.mode == "all":
        for mode in ["protein", "rna", "rna_protein"]:
            emb_path = DEFAULT_EMBEDDING_PATHS[mode]
            tasks.append((mode, emb_path, base_out_dir / mode))
    else:
        mode = args.mode
        emb_path = DEFAULT_EMBEDDING_PATHS[mode]
        tasks.append((mode, emb_path, base_out_dir / mode))

    print("=" * 75)
    print("         MIMIC Ridge Regression: hIPSC_CM Evaluation Pipeline          ")
    print("=" * 75)
    print(f"Modes to run:        {[t[0] for t in tasks]}")
    print(f"Base Output Dir:     {base_out_dir}")
    print(f"Splits Lookup Table: {lookup_path}")
    print(f"Target column:       {args.target_col}")

    summary_records = []
    for mode_name, emb_path, mode_out_dir in tasks:
        metrics = train_and_evaluate_mode(
            mode_name=mode_name,
            emb_path=emb_path,
            lookup_path=lookup_path,
            out_dir=mode_out_dir,
            target_col=args.target_col,
            random_state=args.random_state,
            plot=args.plot,
        )
        summary_records.append({
            "mode": mode_name,
            "best_alpha": metrics["best_alpha"],
            "cv_mean_pearson_r": metrics["cv_mean_pearson_r"],
            "cv_std_pearson_r": metrics["cv_std_pearson_r"],
            "test_pearson_r": metrics.get("test_pearson_r", np.nan),
            "test_spearman_rho": metrics.get("test_spearman_rho", np.nan),
            "test_rmse": metrics.get("test_rmse", np.nan),
            "test_r2": metrics.get("test_r2", np.nan),
        })

    # Print summary table if multiple modes were run
    if len(summary_records) > 1:
        print("\n" + "=" * 80)
        print("                  MIMIC EMBEDDINGS COMPARISON SUMMARY                  ")
        print("=" * 80)
        print(f"{'Mode':<15} | {'Alpha':<9} | {'CV Pearson r (Mean±Std)':<24} | {'Test r':<8} | {'Test rho':<8} | {'Test RMSE':<8}")
        print("-" * 80)
        for s in summary_records:
            cv_str = f"{s['cv_mean_pearson_r']:.4f} ± {s['cv_std_pearson_r']:.4f}"
            print(f"{s['mode']:<15} | {s['best_alpha']:<9.1e} | {cv_str:<24} | {s['test_pearson_r']:>7.4f} | {s['test_spearman_rho']:>8.4f} | {s['test_rmse']:>9.4f}")
        print("=" * 80)

    print("\n[✓] All requested MIMIC Ridge Regression evaluations successfully completed!")


if __name__ == "__main__":
    main()
