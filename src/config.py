"""
config.py
---------
YAML config loader for the pipeline.

Six sections (dataset, schema, cleaning, features, target, modeling)
map to dataclasses. Missing fields fall back to dataclass defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger  # type: ignore

LOG = get_logger("config")


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DatasetCfg:
    name: str = "dataset"
    loader: str = "csv"                       # 'csv', 'excel', 'parquet'
    path: str = ""
    download_url: str | None = None
    excel_sheets: Any = "all"                 # 'all' or list[str]


@dataclass
class SchemaCfg:
    customer_id: str = "customer_id"
    transaction_id: str = "transaction_id"
    transaction_date: str = "transaction_date"
    quantity: str | None = None
    unit_price: str | None = None
    revenue: str | None = None                # if null, computed quantity*unit_price


@dataclass
class CancellationCfg:
    enabled: bool = False
    method: str = "invoice_prefix"            # 'invoice_prefix' or 'negative_quantity'
    prefix: str = "C"


@dataclass
class OutliersCfg:
    method: str = "iqr"                       # 'iqr', 'zscore', 'none'
    k: float = 1.5
    columns: list[str] = field(default_factory=lambda: ["quantity", "unit_price"])


@dataclass
class CleaningCfg:
    drop_cancellations: CancellationCfg = field(default_factory=CancellationCfg)
    drop_missing_customer: bool = True
    drop_nonpositive: bool = True
    drop_duplicates: bool = True
    outliers: OutliersCfg = field(default_factory=OutliersCfg)


@dataclass
class FeaturesCfg:
    log_transform: bool = True
    scaler: str = "standard"                  # 'standard', 'robust', 'minmax', 'none'
    snapshot_date: str = "auto"


@dataclass
class TargetCfg:
    type: str = "top_quantile"                # 'top_quantile', 'threshold', 'top_n'
    quantile: float = 0.75
    threshold: float | None = None
    top_n: int | None = None


@dataclass
class AutoencoderCfg:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3


@dataclass
class ModelingCfg:
    test_size: float = 0.25
    smote: bool = True
    cv_folds: int = 5
    classifiers: list[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "mlp"]
    )
    cluster_models: list[str] = field(
        default_factory=lambda: ["kmeans", "dbscan", "gmm", "ae_kmeans"]
    )
    k_range: list[int] = field(default_factory=lambda: [2, 11])
    dbscan_eps: float = 0.6
    dbscan_min_samples: int = 10
    autoencoder: AutoencoderCfg = field(default_factory=AutoencoderCfg)


@dataclass
class OutputCfg:
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"


@dataclass
class PipelineConfig:
    dataset: DatasetCfg
    schema: SchemaCfg
    cleaning: CleaningCfg
    features: FeaturesCfg
    target: TargetCfg
    modeling: ModelingCfg
    output: OutputCfg

    def canonical_columns(self) -> list[str]:
        """Canonical column names used internally after schema mapping."""
        return [
            "customer_id", "transaction_id", "transaction_date",
            "quantity", "unit_price", "revenue",
        ]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _build(cls, data: dict) -> Any:
    """Construct a dataclass from a dict, recursing into nested dataclasses."""
    if data is None:
        return cls()
    kwargs = {}
    type_hints = {f.name: f.type for f in fields(cls)}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        # Recurse for nested dataclasses (detected by capitalised type hint)
        hint = type_hints[f.name]
        if isinstance(hint, str) and hint.endswith("Cfg"):
            sub_cls = globals()[hint]
            kwargs[f.name] = _build(sub_cls, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> PipelineConfig:
    """Read a YAML file and return a validated PipelineConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    LOG.info("Loading config from %s", path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = PipelineConfig(
        dataset=_build(DatasetCfg, raw.get("dataset", {})),
        schema=_build(SchemaCfg, raw.get("schema", {})),
        cleaning=_build(CleaningCfg, raw.get("cleaning", {})),
        features=_build(FeaturesCfg, raw.get("features", {})),
        target=_build(TargetCfg, raw.get("target", {})),
        modeling=_build(ModelingCfg, raw.get("modeling", {})),
        output=_build(OutputCfg, raw.get("output", {})),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: PipelineConfig) -> None:
    """Sanity checks on config values."""
    if cfg.dataset.loader not in {"csv", "excel", "parquet"}:
        raise ValueError(
            f"dataset.loader must be one of csv/excel/parquet, "
            f"got {cfg.dataset.loader!r}"
        )
    if cfg.features.scaler not in {"standard", "robust", "minmax", "none"}:
        raise ValueError(f"features.scaler invalid: {cfg.features.scaler!r}")
    if cfg.cleaning.outliers.method not in {"iqr", "zscore", "none"}:
        raise ValueError(
            f"cleaning.outliers.method invalid: {cfg.cleaning.outliers.method!r}"
        )
    if cfg.target.type not in {"top_quantile", "threshold", "top_n"}:
        raise ValueError(f"target.type invalid: {cfg.target.type!r}")
    if cfg.target.type == "threshold" and cfg.target.threshold is None:
        raise ValueError("target.type=threshold requires target.threshold")
    if cfg.target.type == "top_n" and cfg.target.top_n is None:
        raise ValueError("target.type=top_n requires target.top_n")
    if cfg.schema.revenue is None and (
        cfg.schema.quantity is None or cfg.schema.unit_price is None
    ):
        raise ValueError(
            "schema must define either revenue, OR both quantity and unit_price"
        )


if __name__ == "__main__":
    cfg = load_config(Path(__file__).resolve().parent.parent
                      / "configs" / "online_retail_ii.yaml")
    LOG.info("Loaded config OK: %s", cfg.dataset.name)
