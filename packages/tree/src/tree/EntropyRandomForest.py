from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

from .DecisionTreeProtocol import DecisionTreeProtocol

class EntropyRandomForest(DecisionTreeProtocol):
    random_state = 42

    @dataclass
    class FitContext:
        category_map: Dict[int, Any]
        n_features: int = 0
        classes: np.ndarray = None

        def decode(self, numeric_array) -> pd.Series:
            """
            Convert numeric predictions (integers) back to category labels.
            numeric_array can be a scalar, list, NumPy array, or pandas Series.
            """
            arr = np.asarray(numeric_array)
            return pd.Series(
                np.array([self.category_map[int(i)] for i in arr]),
                dtype="category",
            )

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth:int=3,
        min_samples_split: int = 2,
        max_features: Optional[int] = None,
        balance_classes: Optional[bool] = None,
        random_split: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.balance_classes = balance_classes
        self.random_split = random_split

        self.ensemble: list[DecisionTreeClassifier] = []
        self.fit_context : Optional["EntropyRandomForest.FitContext"] = None

    # ----------------- training ----------------- #
    def fit(self, x_in: pd.DataFrame , y_in: pd.Series):
        if not y_in.dtype == "category":
            raise TypeError("y must be categorical")

        x = np.asarray(x_in, dtype=float)
        y = np.asarray(y_in.cat.codes, dtype=int)
        category_map = dict(enumerate(y_in.cat.categories))

        n_samples, n_features = x.shape

        if self.max_features is not None and self.max_features > n_features:
            raise ValueError("max_features cannot be higher than n_features")

        self.fit_context = EntropyRandomForest.FitContext(
            category_map=category_map,
            n_features=n_features,
            classes=np.unique(y),
        )
        self.ensemble = []

        rng = np.random.default_rng(EntropyRandomForest.random_state)

        for i in range(self.n_estimators):
            sample_indices = rng.integers(0, n_samples, size=n_samples) # bootstrap indices
            x_boot = x[sample_indices]
            y_boot = y[sample_indices]

            tree = DecisionTreeClassifier(
                criterion="entropy",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
                class_weight="balanced" if self.balance_classes else None,
                splitter="random" if self.random_split else "best",
            )
            tree.fit(x_boot, y_boot)
            self.ensemble.append(tree)
        return self

    def predict(self, x_in: pd.DataFrame):
        if self.fit_context is None:
            raise RuntimeError("The forest has not been fitted yet.")

        x = np.asarray(x_in, dtype=float)
        n_samples = x.shape[0]

        assert len(self.ensemble) == self.n_estimators
        preds_by_tree = np.empty((self.n_estimators, n_samples), dtype=object)
        for i, tree in enumerate(self.ensemble):
            preds_by_tree[i, :] = tree.predict(x)

        y_pred = np.empty(n_samples, dtype=object) # store majority votes per sample
        for j in range(n_samples):
            vals, counts = np.unique(preds_by_tree[:, j], return_counts=True)
            y_pred[j] = vals[np.argmax(counts)] # assign that pred that has the index for the max count

        return self.fit_context.decode(y_pred)
