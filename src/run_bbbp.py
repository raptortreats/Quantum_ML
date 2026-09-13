"""Curate BBBP, fit the Aer QSVC, and write README / notebook figures."""

from __future__ import annotations

from pathlib import Path

from src.bbbp_data import curate_bbbp_subset, save_subset_csv
from src.bbbp_plots import (
    save_confusion_matrix,
    save_feature_map,
    save_kernel_embedding,
    save_kernel_heatmaps,
    save_metrics_table,
    save_molecule_grid,
    save_prediction_examples,
    save_prediction_table,
    save_roc_curve,
    save_similarity_maps,
)
from src.qkernel import fit_quantum_kernel_qsvc, nearest_train_neighbors

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def run(figures_dir: Path = FIGURES, data_dir: Path = DATA) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    bundle = curate_bbbp_subset(data_dir)
    save_subset_csv(bundle, data_dir / "bbbp_curated.csv")

    fit = fit_quantum_kernel_qsvc(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        seed=int(bundle.curation["seed"]),
    )
    neighbor_idx, _ = nearest_train_neighbors(fit.k_test, bundle.y_train)

    paths = {
        "feature_map": save_feature_map(fit.kernel.feature_map, figures_dir / "bbbp_feature_map.png"),
        "kernel_heatmaps": save_kernel_heatmaps(
            fit.k_train, fit.k_test, bundle.y_train, bundle.y_test, figures_dir / "bbbp_kernel_heatmaps.png"
        ),
        "roc": save_roc_curve(fit.metrics, figures_dir / "bbbp_roc.png"),
        "confusion": save_confusion_matrix(fit.metrics, figures_dir / "bbbp_confusion_matrix.png"),
        "metrics": save_metrics_table(fit.metrics, fit.classical_metrics, figures_dir / "bbbp_metrics_table.png"),
        "embedding": save_kernel_embedding(fit.k_train, bundle.y_train, figures_dir / "bbbp_kernel_embedding.png"),
        "examples": save_prediction_examples(
            bundle.test_frame, bundle.y_test, fit.y_pred, fit.y_score, figures_dir / "bbbp_prediction_examples.png"
        ),
        "pred_table": save_prediction_table(
            bundle.test_frame, bundle.y_test, fit.y_pred, fit.y_score, figures_dir / "bbbp_prediction_table.png"
        ),
        "similarity": save_similarity_maps(
            bundle.test_frame,
            bundle.train_frame,
            bundle.y_test,
            fit.y_pred,
            fit.y_score,
            neighbor_idx,
            figures_dir / "bbbp_similarity_maps.png",
        ),
        "mol_grid": save_molecule_grid(bundle.frame, bundle.y, figures_dir / "bbbp_mol_grid.png"),
    }

    print("curation:", {k: bundle.curation[k] for k in (
        "n_raw", "n_valid", "n_confident", "n_keep", "n_train", "n_test", "pca_explained_variance_sum"
    )})
    print("QSVC:", {k: fit.metrics[k] for k in ("accuracy", "auc", "f1", "precision", "recall", "n_correct")})
    print("RBF :", {k: fit.classical_metrics[k] for k in ("accuracy", "auc", "f1")})
    return {"bundle": bundle, "fit": fit, "paths": paths}


if __name__ == "__main__":
    run()
