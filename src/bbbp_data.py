"""Load, curate, and featurize a simulator-sized BBBP subset.

The public MoleculeNet / DeepChem BBBP file has ~2 050 compounds. A 4-qubit
Aer kernel cannot honestly represent that full fingerprint space, so this
module builds a **documented, stratified subset** (~100 molecules):

1. Drop rows whose SMILES RDKit cannot parse (11 in the standard file).
2. Canonicalize and de-duplicate SMILES.
3. Compute Morgan fingerprints (radius 2, 2048 bits).
4. Rank bits by mutual information with ``p_np`` and fit a cheap balanced
   logistic regression on the top pool (default 24 bits).
5. Keep only high-confidence molecules (predicted P close to 0 or 1).
6. Draw a balanced sample (50 penetrant / 50 non-penetrant) with a fixed seed.
7. Stratified train/test split, then **re-select bits and fit PCA on train
   only** so the 4-qubit features do not leak the test labels.

Step 4–6 are *subset construction*, not the quantum model. They enrich the
demo for class contrast on a laptop simulator. They are **not** the MoleculeNet
scaffold-split leaderboard protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

RDLogger.DisableLog("rdApp.*")

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
BBBP_FILENAME = "BBBP.csv"

SEED = 21
N_KEEP = 100
TEST_SIZE = 0.25
N_BIT_POOL = 24
N_FEATURES = 4
CONFIDENCE_THRESHOLD = 0.15
FP_RADIUS = 2
FP_SIZE = 2048
FEATURE_SCALE = float(np.pi)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def ensure_bbbp_csv(data_dir: Path | None = None, url: str = BBBP_URL) -> Path:
    """Return the local MoleculeNet BBBP CSV, downloading it if needed."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / BBBP_FILENAME
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(url, path)
    return path


def _morgan_generator(radius: int = FP_RADIUS, fp_size: int = FP_SIZE):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def morgan_fingerprint(mol: Chem.Mol, radius: int = FP_RADIUS, fp_size: int = FP_SIZE) -> np.ndarray:
    return np.asarray(_morgan_generator(radius, fp_size).GetFingerprint(mol), dtype=np.float64)


def load_valid_bbbp(data_dir: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse BBBP, dropping invalid SMILES and duplicate canonical structures."""
    path = ensure_bbbp_csv(data_dir)
    raw = pd.read_csv(path)
    required = {"name", "p_np", "smiles"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    generator = _morgan_generator()
    rows: list[dict] = []
    fingerprints: list[np.ndarray] = []
    seen: set[str] = set()
    n_invalid = 0
    n_duplicate = 0

    for rec in raw.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(rec.smiles))
        if mol is None:
            n_invalid += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            n_duplicate += 1
            continue
        seen.add(canonical)
        rows.append(
            {
                "name": str(rec.name),
                "smiles": str(rec.smiles),
                "canonical_smiles": canonical,
                "label": int(rec.p_np),
            }
        )
        fingerprints.append(np.asarray(generator.GetFingerprint(mol), dtype=np.float64))

    frame = pd.DataFrame(rows)
    info = {
        "source_path": str(path),
        "n_raw": int(len(raw)),
        "n_invalid_smiles": int(n_invalid),
        "n_duplicate_smiles": int(n_duplicate),
        "n_valid": int(len(frame)),
        "n_positive": int(frame["label"].sum()),
        "n_negative": int((1 - frame["label"]).sum()),
    }
    frame.attrs["load_info"] = info
    return frame, np.vstack(fingerprints)


def high_confidence_indices(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    *,
    n_bit_pool: int = N_BIT_POOL,
    threshold: float = CONFIDENCE_THRESHOLD,
    seed: int = SEED,
) -> tuple[np.ndarray, dict]:
    """Molecules a cheap fingerprint logistic model already considers clear-cut."""
    mi = mutual_info_classif(fingerprints, labels, random_state=seed)
    bits = np.argsort(mi)[-n_bit_pool:]
    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
    clf.fit(fingerprints[:, bits], labels)
    proba = clf.predict_proba(fingerprints[:, bits])[:, 1]
    mask = (proba <= threshold) | (proba >= 1.0 - threshold)
    idx = np.flatnonzero(mask)
    report = {
        "n_bit_pool": int(n_bit_pool),
        "threshold": float(threshold),
        "n_confident": int(mask.sum()),
        "n_confident_positive": int((labels[idx] == 1).sum()),
        "n_confident_negative": int((labels[idx] == 0).sum()),
        "bit_indices_for_curation": bits.astype(int).tolist(),
    }
    return idx, report


def balanced_sample(
    labels: np.ndarray,
    candidate_idx: np.ndarray,
    *,
    n_keep: int = N_KEEP,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = candidate_idx[labels[candidate_idx] == 1]
    neg = candidate_idx[labels[candidate_idx] == 0]
    n_each = min(n_keep // 2, len(pos), len(neg))
    if n_each < 8:
        raise ValueError(
            f"Not enough confident molecules to build a balanced subset "
            f"(pos={len(pos)}, neg={len(neg)})."
        )
    selected = np.concatenate(
        [rng.choice(pos, n_each, replace=False), rng.choice(neg, n_each, replace=False)]
    )
    rng.shuffle(selected)
    return selected


@dataclass
class FeatureCompressor:
    """Train-only MI bit selection + PCA down to ``n_features`` angles in ``[0, π]``."""

    n_bit_pool: int = N_BIT_POOL
    n_features: int = N_FEATURES
    seed: int = SEED
    bit_indices_: np.ndarray | None = None
    pca_: PCA | None = None
    scaler_: MinMaxScaler | None = None
    explained_variance_ratio_: np.ndarray | None = None

    def fit(self, fingerprints: np.ndarray, labels: np.ndarray) -> FeatureCompressor:
        mi = mutual_info_classif(fingerprints, labels, random_state=self.seed)
        self.bit_indices_ = np.argsort(mi)[-self.n_bit_pool:]
        pooled = fingerprints[:, self.bit_indices_]
        self.pca_ = PCA(n_components=self.n_features, random_state=self.seed)
        reduced = self.pca_.fit_transform(pooled)
        self.scaler_ = MinMaxScaler(feature_range=(0.0, FEATURE_SCALE))
        self.scaler_.fit(reduced)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_.copy()
        return self

    def transform(self, fingerprints: np.ndarray) -> np.ndarray:
        if self.bit_indices_ is None or self.pca_ is None or self.scaler_ is None:
            raise RuntimeError("FeatureCompressor must be fit before transform().")
        reduced = self.pca_.transform(fingerprints[:, self.bit_indices_])
        return self.scaler_.transform(reduced)

    def fit_transform(self, fingerprints: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(fingerprints, labels).transform(fingerprints)


@dataclass
class BBBPSubset:
    """Curated BBBP bundle used by the notebook and figure script."""

    frame: pd.DataFrame
    fingerprints: np.ndarray
    features: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    compressor: FeatureCompressor
    curation: dict = field(default_factory=dict)

    @property
    def X_train(self) -> np.ndarray:
        return self.features[self.train_idx]

    @property
    def X_test(self) -> np.ndarray:
        return self.features[self.test_idx]

    @property
    def y_train(self) -> np.ndarray:
        return self.y[self.train_idx]

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.test_idx]

    @property
    def train_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.train_idx].reset_index(drop=True)

    @property
    def test_frame(self) -> pd.DataFrame:
        return self.frame.iloc[self.test_idx].reset_index(drop=True)


def curate_bbbp_subset(
    data_dir: Path | None = None,
    *,
    n_keep: int = N_KEEP,
    test_size: float = TEST_SIZE,
    n_bit_pool: int = N_BIT_POOL,
    n_features: int = N_FEATURES,
    threshold: float = CONFIDENCE_THRESHOLD,
    seed: int = SEED,
) -> BBBPSubset:
    frame, fingerprints = load_valid_bbbp(data_dir)
    labels = frame["label"].to_numpy()
    load_info = dict(frame.attrs.get("load_info", {}))

    confident_idx, confident_report = high_confidence_indices(
        fingerprints,
        labels,
        n_bit_pool=n_bit_pool,
        threshold=threshold,
        seed=seed,
    )
    selected = balanced_sample(labels, confident_idx, n_keep=n_keep, seed=seed)

    subset = frame.iloc[selected].reset_index(drop=True)
    subset_fp = fingerprints[selected]
    subset_y = subset["label"].to_numpy()

    train_idx, test_idx = train_test_split(
        np.arange(len(subset)),
        test_size=test_size,
        stratify=subset_y,
        random_state=seed,
    )
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    compressor = FeatureCompressor(n_bit_pool=n_bit_pool, n_features=n_features, seed=seed)
    compressor.fit(subset_fp[train_idx], subset_y[train_idx])
    features = compressor.transform(subset_fp)

    split = np.full(len(subset), "train", dtype=object)
    split[test_idx] = "test"
    subset = subset.copy()
    subset["split"] = split

    curation = {
        **load_info,
        **confident_report,
        "n_keep": int(len(subset)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int(subset_y[train_idx].sum()),
        "n_train_negative": int((1 - subset_y[train_idx]).sum()),
        "n_test_positive": int(subset_y[test_idx].sum()),
        "n_test_negative": int((1 - subset_y[test_idx]).sum()),
        "seed": int(seed),
        "test_size": float(test_size),
        "n_features": int(n_features),
        "pca_explained_variance_ratio": compressor.explained_variance_ratio_.tolist(),
        "pca_explained_variance_sum": float(np.sum(compressor.explained_variance_ratio_)),
        "feature_scale": FEATURE_SCALE,
        "fp_radius": FP_RADIUS,
        "fp_size": FP_SIZE,
        "notes": (
            "Balanced high-confidence subset for a 4-qubit Aer demo. "
            "Not a MoleculeNet scaffold-split leaderboard result."
        ),
    }
    return BBBPSubset(
        frame=subset,
        fingerprints=subset_fp,
        features=features,
        y=subset_y,
        train_idx=train_idx,
        test_idx=test_idx,
        compressor=compressor,
        curation=curation,
    )


def save_subset_csv(bundle: BBBPSubset, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle.frame.to_csv(path, index=False)
    return path
