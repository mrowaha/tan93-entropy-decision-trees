from dataclasses import dataclass, InitVar
from typing import Self, Optional, Dict, Any

import numpy as np
import pandas as pd
from graphviz import Digraph

from utils.datasets import (
    get_information_gain_by_n,
    get_class_counts,
)

from .DecisionTreeProtocol import DecisionTreeProtocol

class EntropyDecisionTree(DecisionTreeProtocol):

    @dataclass(frozen=True)
    class BestSplit:
        feature_index: int
        feature_name: str
        threshold: float
        left_split_mask: np.ndarray
        right_split_mask: np.ndarray

        def to_dict(self):
            return {
                "feature_index": self.feature_index,
                "threshold": self.threshold
            }

    @dataclass
    class Node:
        best_split: "EntropyDecisionTree.BestSplit" = None
        left: "EntropyDecisionTree.Node" = None
        right: "EntropyDecisionTree.Node" = None
        prediction: int = None
        prediction_category: Any = None
        class_counts: np.ndarray = None
        depth: int = 0

        def visualize(self) -> Digraph:
            """
             Convert your custom decision tree into a GraphViz Digraph.
             feature_names: optional list mapping feature indices → names.
             """
            dot = Digraph()
            dot.attr("node", shape="box", style="filled", color="lightgray")

            def add_node(node: "EntropyDecisionTree.Node", node_id):
                if node is None:
                    return

                if node.best_split is None:
                    label = f"Leaf\ncategory={node.prediction_category}\ncounts={node.class_counts}"
                    dot.node(node_id, label=label, color="lightblue")
                    return

                bs = node.best_split
                label = f"{bs.feature_name} ≤ {bs.threshold:.4f}\n"
                dot.node(node_id, label=label, color="lightgreen")
                if node.left:
                    left_id = f"{node_id}L"
                    dot.edge(node_id, left_id, label="True")
                    add_node(node.left, left_id)

                if node.right:
                    right_id = f"{node_id}R"
                    dot.edge(node_id, right_id, label="False")
                    add_node(node.right, right_id)

            add_node(self, "0")
            return dot

        def to_dict(self):
            return {
                "best_split": self.best_split.to_dict() if self.best_split else None,
                "prediction": self.prediction,
                "prediction_category": self.prediction_category,
                "class_counts": (
                    self.class_counts.tolist()
                    if isinstance(self.class_counts, np.ndarray)
                    else self.class_counts
                ),
                "left": self.left.to_dict() if self.left else None,
                "right": self.right.to_dict() if self.right else None,
            }

    @dataclass
    class FitContext:
        y: InitVar[np.ndarray]
        category_map: Dict[int, Any]
        feature_map: Dict[int, str]
        n_classes: int = 0

        def __post_init__(self, y):
            self.n_classes = len(np.unique(y))

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

    def __init__(self: Self, max_depth: int = 3, min_samples_split: int = 2):
        self._max_depth: int = max_depth
        self._min_samples_split: int = min_samples_split

        self.root : Optional["EntropyDecisionTree.Node"] = None
        self.fit_context : Optional["EntropyDecisionTree.FitContext"] = None

    @property
    def name(self):
        return "entropy"

    @property
    def max_depth(self: Self) -> int:
        return self._max_depth

    @property
    def min_samples_split(self: Self) -> int:
        return self._min_samples_split

# public usage
    def fit(self, x_in: pd.DataFrame, y_in: pd.Series):
        if not y_in.dtype == "category":
            raise TypeError("y must be categorical")

        x = np.asarray(x_in, dtype=float)
        y = np.asarray(y_in.cat.codes, dtype=int)
        category_map = dict(enumerate(y_in.cat.categories))
        feature_map = dict(enumerate(x_in.columns))
        self.fit_context = EntropyDecisionTree.FitContext(y=y, category_map=category_map, feature_map=feature_map)
        self.root = self._fit_classifier_tree_recursively(x, y, depth=0)

    def predict(self, x_in: pd.DataFrame):
        if self.root is None:
            raise RuntimeError("fit model first to predict")
        x = np.asarray(x_in, dtype=float)
        y_pred = np.array([self._predict_one(_x, self.root) for _x in x])
        return self.fit_context.decode(y_pred)

# helpers
    def _fit_classifier_tree_recursively(self: Self, x, y, depth: int) -> Node:
        node = EntropyDecisionTree.Node(depth=depth)
        node.class_counts = get_class_counts(self.fit_context.n_classes, y)

        # base condition
        if depth >= self._max_depth \
                or len(np.unique(y)) == 1 \
                or len(y) < self._min_samples_split:
            node.prediction = int(np.argmax(node.class_counts))
            node.prediction_category = self.fit_context.category_map[node.prediction]
            return node

        best_split = self._generate_best_split(x, y)
        if best_split is None:
            node.prediction = int(np.argmax(node.class_counts))
            node.prediction_category = self.fit_context.category_map[node.prediction]
            return node

        node.best_split = best_split
        node.left = self._fit_classifier_tree_recursively(x[best_split.left_split_mask], y[best_split.left_split_mask], depth=depth+1)
        node.right = self._fit_classifier_tree_recursively(x[best_split.right_split_mask], y[best_split.right_split_mask], depth=depth+1)
        return node

    def _generate_best_split(self: Self, x, y) -> Optional[BestSplit]:
        n_samples, n_feature_count = x.shape
        if n_samples < self.min_samples_split:
            return None

        calculate_ig = get_information_gain_by_n(n_samples)
        best_ig : float = 0.0
        best_feature_index : Optional[int] = None
        best_threshold: Optional[float] = None
        best_left_split: Optional[np.ndarray] = None
        best_right_split: Optional[np.ndarray] = None


        for feature_index in range(n_feature_count):
            x_feature_index : np.ndarray = x[:, feature_index]
#             sort both the current inputs for a given feature and the corresponding labels. need sorts to find candidate thresholds for a split
            sort_inputs_by = np.argsort(x_feature_index)
            x_feature_index_sorted, y_sorted  = x_feature_index[sort_inputs_by], y[sort_inputs_by]

            # all candidate thresholds
            diff_mask = np.diff(x_feature_index_sorted) != 0
            thresholds = 0.5 * (x_feature_index_sorted[:-1][diff_mask] + x_feature_index_sorted[1:][diff_mask])
            for t in thresholds:
                left_mask = x_feature_index_sorted <= t
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    # it is a trivial split if either of the masks/splits has zero samples
                    continue

                ig = calculate_ig(y_sorted, left_mask, right_mask)
                if ig > best_ig:
                    best_ig = ig
                    best_feature_index = feature_index
                    best_threshold = t
                    best_left_split = x_feature_index <= t
                    best_right_split = ~best_left_split

        if best_ig <= 0.0:
            return None

        return EntropyDecisionTree.BestSplit(
            feature_index=best_feature_index,
            feature_name=self.fit_context.feature_map[best_feature_index],
            threshold=best_threshold,
            left_split_mask=best_left_split,
            right_split_mask=best_right_split
        )

    def _predict_one(self, x, node: Node) -> Any:
        best_split = node.best_split
        # If leaf or no split information
        if node.best_split is None:
            return node.prediction

        if x[best_split.feature_index] <= best_split.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)



