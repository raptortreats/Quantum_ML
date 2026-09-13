"""Publication-style figures for the BBBP quantum-kernel notebook."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import KernelPCA

INK = "#102a43"
TEAL = "#0d9488"
CORAL = "#c23b22"
SLATE = "#486581"
PAPER = "#f8fafc"
OK = "#0f766e"
BAD = "#b91c1c"

KERNEL_CMAP = LinearSegmentedColormap.from_list(
    "bbbp_kernel", ["#f8fafc", "#99f6e4", "#0d9488", "#115e59"]
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PAPER,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
            "axes.titleweight": "semibold",
            "axes.grid": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 160,
        }
    )


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _order_by_label(labels: np.ndarray) -> np.ndarray:
    return np.argsort(labels, kind="stable")


def save_feature_map(feature_map, path: Path) -> Path:
    apply_style()
    fig = feature_map.draw("mpl", style="clifford")
    fig.set_size_inches(11.4, 3.4)
    fig.suptitle("4-qubit ZZ feature map  ·  Aer statevector input", color=INK, y=1.04)
    return _save(fig, path)


def save_kernel_heatmaps(
    k_train: np.ndarray,
    k_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    path: Path,
) -> Path:
    apply_style()
    tr = _order_by_label(y_train)
    te = _order_by_label(y_test)
    k_tt = k_train[np.ix_(tr, tr)]
    k_te = k_test[np.ix_(te, tr)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    im0 = axes[0].imshow(k_tt, cmap=KERNEL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    axes[0].set_title("Train–train kernel  K(xᵢ, xⱼ)")
    axes[0].set_xlabel("train (sorted by label)")
    axes[0].set_ylabel("train (sorted by label)")
    im1 = axes[1].imshow(k_te, cmap=KERNEL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    axes[1].set_title("Test–train kernel  K(x*, xⱼ)")
    axes[1].set_xlabel("train (sorted by label)")
    axes[1].set_ylabel("test (sorted by label)")
    for ax, labels, axis in (
        (axes[0], y_train[tr], "x"),
        (axes[0], y_train[tr], "y"),
        (axes[1], y_train[tr], "x"),
        (axes[1], y_test[te], "y"),
    ):
        cuts = np.where(np.diff(labels) != 0)[0] + 0.5
        for cut in cuts:
            if axis == "x":
                ax.axvline(cut, color=INK, lw=0.7, alpha=0.35)
            else:
                ax.axhline(cut, color=INK, lw=0.7, alpha=0.35)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("fidelity")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("fidelity")
    fig.tight_layout()
    return _save(fig, path)


def save_roc_curve(metrics: dict, path: Path, title: str = "QSVC ROC on the curated test split") -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(metrics["fpr"], metrics["tpr"], color=TEAL, lw=2.4, label=f"QSVC  AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1.0, label="chance")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def save_confusion_matrix(metrics: dict, path: Path) -> Path:
    apply_style()
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=KERNEL_CMAP, vmin=0)
    labels = ["non-penetrant (0)", "penetrant (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color=INK, fontsize=16, fontweight="semibold")
    fig.tight_layout()
    return _save(fig, path)


def save_metrics_table(q_metrics: dict, classical_metrics: dict, path: Path) -> Path:
    apply_style()
    keys = ["accuracy", "auc", "f1", "precision", "recall"]
    cell = [
        ["QSVC (quantum kernel)", *[f"{q_metrics[k]:.3f}" for k in keys]],
        ["RBF SVM (same 4 features)", *[f"{classical_metrics[k]:.3f}" for k in keys]],
    ]
    fig, ax = plt.subplots(figsize=(9.6, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell,
        colLabels=["model", *keys],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.7)
    for (row, col), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#cbd5e1")
        if row == 0:
            cell_obj.set_facecolor(TEAL)
            cell_obj.set_text_props(color="white", fontweight="semibold")
        elif row == 1:
            cell_obj.set_facecolor("#ccfbf1")
        else:
            cell_obj.set_facecolor(PAPER)
    ax.set_title("Test metrics on the curated 25-molecule hold-out  (not MoleculeNet)", pad=12)
    fig.tight_layout()
    return _save(fig, path)


def save_kernel_embedding(k_train: np.ndarray, y_train: np.ndarray, path: Path) -> Path:
    apply_style()
    embed = KernelPCA(n_components=2, kernel="precomputed", random_state=21)
    xy = embed.fit_transform(k_train)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for label, color, name in ((0, CORAL, "non-penetrant"), (1, TEAL, "penetrant")):
        mask = y_train == label
        ax.scatter(xy[mask, 0], xy[mask, 1], c=color, s=42, alpha=0.9, label=name, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("kernel PCA 1")
    ax.set_ylabel("kernel PCA 2")
    ax.set_title("Train embedding from the Aer fidelity kernel")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, path)


def _mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _mol_png(smiles: str, size: tuple[int, int] = (260, 200)) -> PILImage.Image:
    return Draw.MolToImage(_mol(smiles), size=size)


def save_prediction_examples(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    path: Path,
    n_show: int = 8,
) -> Path:
    apply_style()
    table = test_frame.copy()
    table["true"] = y_true
    table["pred"] = y_pred
    table["score"] = y_score
    table["correct"] = table["true"] == table["pred"]
    table = table.sort_values(["correct", "score"], ascending=[True, False])
    show = table.head(n_show)

    n = len(show)
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n + 1.2, 5.6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i, rec in enumerate(show.itertuples(index=False)):
        axes[0, i].imshow(_mol_png(rec.smiles, size=(240, 180)))
        axes[0, i].axis("off")
        mark = "correct" if rec.correct else "error"
        color = OK if rec.correct else BAD
        axes[0, i].set_title(f"{mark}", color=color, fontsize=9)
        axes[1, i].axis("off")
        label = "P" if rec.true == 1 else "NP"
        pred = "P" if rec.pred == 1 else "NP"
        mol_name = str(rec[0])[:18]
        axes[1, i].text(
            0.5,
            0.7,
            f"{mol_name}\ntrue {label}  pred {pred}\nscore {rec.score:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=INK,
            transform=axes[1, i].transAxes,
        )
    fig.suptitle("Held-out predictions  ·  P = penetrant, NP = non-penetrant", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def save_prediction_table(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    path: Path,
) -> Path:
    apply_style()
    rows = []
    for rec, yt, yp, ys in zip(test_frame.itertuples(index=False), y_true, y_pred, y_score):
        rows.append(
            [
                str(rec[0])[:22],
                "penetrant" if yt == 1 else "non-pen.",
                "penetrant" if yp == 1 else "non-pen.",
                f"{ys:+.2f}",
                "yes" if yt == yp else "no",
            ]
        )
    fig, ax = plt.subplots(figsize=(9.8, 0.42 * len(rows) + 1.1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["molecule", "true", "predicted", "decision", "correct"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.15, 1.35)
    for (row, col), cell_obj in table.get_celld().items():
        cell_obj.set_edgecolor("#e2e8f0")
        if row == 0:
            cell_obj.set_facecolor(INK)
            cell_obj.set_text_props(color="white", fontweight="semibold")
        elif row >= 1 and rows[row - 1][4] == "no":
            cell_obj.set_facecolor("#fee2e2")
        else:
            cell_obj.set_facecolor("#f0fdfa" if row % 2 else PAPER)
    ax.set_title("Every test molecule")
    fig.tight_layout()
    return _save(fig, path)


def _similarity_png(ref_smiles: str, probe_smiles: str, size: tuple[int, int] = (360, 280)) -> PILImage.Image:
    ref = _mol(ref_smiles)
    probe = _mol(probe_smiles)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    SimilarityMaps.GetSimilarityMapForFingerprintGenerator(ref, probe, generator, drawer)
    drawer.FinishDrawing()
    return PILImage.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def pick_example_indices(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> list[int]:
    """Prefer two correct and two incorrect test rows, mixed across classes."""
    correct = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    chosen: list[int] = []

    def take(pool: np.ndarray, label: int) -> None:
        hits = pool[y_true[pool] == label]
        if len(hits):
            best = hits[np.argmax(np.abs(y_score[hits]))]
            if int(best) not in chosen:
                chosen.append(int(best))

    for label in (1, 0):
        take(correct, label)
    for label in (1, 0):
        take(wrong, label)
    leftover = [int(i) for i in np.arange(len(y_true)) if int(i) not in chosen]
    while len(chosen) < min(4, len(y_true)) and leftover:
        chosen.append(leftover.pop(0))
    return chosen[:4]


def save_similarity_maps(
    test_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    neighbor_idx: np.ndarray,
    path: Path,
) -> Path:
    apply_style()
    picks = pick_example_indices(y_true, y_pred, y_score)
    n = len(picks)
    fig, axes = plt.subplots(n, 3, figsize=(10.8, 3.15 * n))
    if n == 1:
        axes = np.array([axes])
    for row, idx in enumerate(picks):
        probe = test_frame.iloc[idx]
        neigh = train_frame.iloc[int(neighbor_idx[idx])]
        ok = y_true[idx] == y_pred[idx]
        axes[row, 0].imshow(_mol_png(probe["smiles"], size=(300, 220)))
        axes[row, 1].imshow(_mol_png(neigh["smiles"], size=(300, 220)))
        axes[row, 2].imshow(_similarity_png(neigh["smiles"], probe["smiles"]))
        for ax in axes[row]:
            ax.axis("off")
        tlab = "P" if y_true[idx] == 1 else "NP"
        plab = "P" if y_pred[idx] == 1 else "NP"
        nlab = "P" if int(neigh["label"]) == 1 else "NP"
        status = "correct" if ok else "incorrect"
        axes[row, 0].set_title(
            f"test  {str(probe['name'])[:20]}\ntrue {tlab} · pred {plab} · {status}",
            fontsize=9,
            color=OK if ok else BAD,
        )
        axes[row, 1].set_title(
            f"nearest train  {str(neigh['name'])[:20]}\nlabel {nlab}",
            fontsize=9,
            color=SLATE,
        )
        axes[row, 2].set_title("Morgan similarity map\n(probe vs train neighbor)", fontsize=9, color=INK)
    fig.suptitle("Atom highlights: green / magenta mark bits that raise / lower similarity to the neighbor", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def save_molecule_grid(frame: pd.DataFrame, y: np.ndarray, path: Path, n: int = 12) -> Path:
    apply_style()
    pos = frame.index[y == 1][: n // 2]
    neg = frame.index[y == 0][: n // 2]
    picks = list(neg) + list(pos)
    mols = [_mol(frame.loc[i, "smiles"]) for i in picks]
    legends = [
        f"{frame.loc[i, 'name'][:16]}  ({'P' if y[i] == 1 else 'NP'})" for i in picks
    ]
    img = Draw.MolsToGridImage(
        mols, molsPerRow=4, subImgSize=(200, 160), legends=legends, returnPNG=True
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, (bytes, bytearray)):
        path.write_bytes(img)
    elif hasattr(img, "save"):
        img.save(path)
    else:
        path.write_bytes(getattr(img, "data"))
    return path
