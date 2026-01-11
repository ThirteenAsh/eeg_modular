from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class UMAPBoundaryConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    grid_res: int = 600
    margin: float = 0.5

    svm_kernel: str = "rbf"  # 'rbf' | 'linear'
    svm_C: float = 10.0
    svm_gamma: str = "scale"

    mode: Literal["filled", "lines", "both"] = "both"
    alpha: float = 0.25


def _fit_surrogate_and_predict_grid(X_2d: np.ndarray, y: np.ndarray, xx: np.ndarray, yy: np.ndarray, cfg: UMAPBoundaryConfig) -> np.ndarray:
    from sklearn.svm import SVC

    if cfg.svm_kernel == "rbf":
        clf = SVC(kernel="rbf", C=float(cfg.svm_C), gamma=cfg.svm_gamma)
    elif cfg.svm_kernel == "linear":
        clf = SVC(kernel="linear", C=float(cfg.svm_C))
    else:
        raise ValueError(f"Unknown svm_kernel: {cfg.svm_kernel}")

    clf.fit(X_2d, y)
    grid_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return grid_pred


def save_umap_svm_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    save_path: str,
    cfg: Optional[UMAPBoundaryConfig] = None,
    title: str = "UMAP Projection with Decision Boundary (Test Set)",
) -> Tuple[np.ndarray, np.ndarray]:
    """Project X to 2D using UMAP (fit on X), then fit an SVM surrogate in 2D and plot decision boundary.

    Returns (X_umap, grid_pred) for optional downstream use.

    Requires `umap-learn`. Caller should catch ImportError if not installed.
    """
    cfg = cfg or UMAPBoundaryConfig()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap  # type: ignore

    reducer = umap.UMAP(
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        random_state=int(cfg.random_state),
    )
    X_umap = reducer.fit_transform(X)

    x_min, x_max = X_umap[:, 0].min() - cfg.margin, X_umap[:, 0].max() + cfg.margin
    y_min, y_max = X_umap[:, 1].min() - cfg.margin, X_umap[:, 1].max() + cfg.margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, int(cfg.grid_res)),
        np.linspace(y_min, y_max, int(cfg.grid_res)),
    )

    grid_pred = _fit_surrogate_and_predict_grid(X_umap, y, xx, yy, cfg)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.Set1

    if cfg.mode in ("filled", "both"):
        plt.contourf(xx, yy, grid_pred, alpha=float(cfg.alpha), cmap=cmap)

    if cfg.mode in ("lines", "both"):
        n_cls = len(np.unique(grid_pred))
        if n_cls >= 2:
            levels = np.arange(0.5, n_cls, 1.0)
            cs = plt.contour(xx, yy, grid_pred, levels=levels, colors="k", linewidths=1.0)
            plt.clabel(cs, inline=True, fontsize=8, fmt="")

    for i, name in enumerate(class_names):
        idxs = (y == i)
        if np.any(idxs):
            plt.scatter(
                X_umap[idxs, 0],
                X_umap[idxs, 1],
                label=name,
                alpha=0.75,
                edgecolor="k",
                s=36,
            )

    plt.legend()
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return X_umap, grid_pred
