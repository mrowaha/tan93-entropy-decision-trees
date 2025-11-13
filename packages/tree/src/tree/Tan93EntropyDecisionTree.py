from dataclasses import dataclass, field
from typing import Dict, Set, Self, Optional,  override

import numpy as np
import pandas as pd

from utils.datasets import (
    get_information_gain_by_n,
    get_class_counts,
)

from .EntropyDecisionTree import EntropyDecisionTree

class Tan93EntropyDecisionTree(EntropyDecisionTree):

    @dataclass
    class Tan93FitContext:
        used_features: Set[str] = field(default_factory=set)

    def __init__(self, feature_costs: Dict[str, float], *,max_depth: int = 3, min_samples_split: int = 2, feature_cost_deps: Optional[Dict[str, Set[str]]] = None):
        super().__init__(max_depth, min_samples_split)
        self.feature_costs = feature_costs
        self.feature_cost_deps = feature_cost_deps
        self.tan93_fit_context: Optional["Tan93EntropyDecisionTree.Tan93FitContext"] = None

    @property
    @override
    def name(self):
        return "tan93+entropy"

    @override
    def fit(self, x_in: pd.DataFrame, y_in: pd.Series):
        self.tan93_fit_context = Tan93EntropyDecisionTree.Tan93FitContext()
        super().fit(x_in, y_in)

    def classification_cost(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        x = np.asarray(x_test, dtype=float)
        y = np.asarray(y_test.cat.codes, dtype=int)

        costs = []
        for i in range(len(x)):
            cost = self._classification_cost_one(x[i], self.root)
            costs.append(cost)

        # average cost per actual class
        results = {}
        for cls in np.unique(y):
            mask = y == cls
            results[self.fit_context.category_map[cls]] = np.mean(np.array(costs)[mask])

        return results

    @override
    def _fit_classifier_tree_recursively(self: Self, x, y, depth: int) -> EntropyDecisionTree.Node:
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

        self.tan93_fit_context.used_features.add(self.fit_context.feature_map[best_split.feature_index])
        node.best_split = best_split
        node.left = self._fit_classifier_tree_recursively(x[best_split.left_split_mask], y[best_split.left_split_mask], depth=depth+1)
        node.right = self._fit_classifier_tree_recursively(x[best_split.right_split_mask], y[best_split.right_split_mask], depth=depth+1)
        return node

    @override
    def _generate_best_split(self: Self, x, y) -> Optional[EntropyDecisionTree.BestSplit]:
        n_samples, n_feature_count = x.shape
        if n_samples < self.min_samples_split:
            return None

        calculate_ig = get_information_gain_by_n(n_samples)
        best_ig: float = 0.0
        best_feature_index: Optional[int] = None
        best_threshold: Optional[float] = None
        best_left_split: Optional[np.ndarray] = None
        best_right_split: Optional[np.ndarray] = None

        for feature_index in range(n_feature_count):
            feature_name = self.fit_context.feature_map[feature_index]
            effective_cost = self.feature_costs.get(feature_name, 1.0)
            if feature_name in self.tan93_fit_context.used_features:
                effective_cost = 1.0

            if self.feature_cost_deps is not None:
                for deps in self.feature_cost_deps.get(feature_name, set()):
                    if deps not in self.tan93_fit_context.used_features:
                        effective_cost += self.feature_costs.get(deps, 1.0)

            x_feature_index: np.ndarray = x[:, feature_index]
            #             sort both the current inputs for a given feature and the corresponding labels. need sorts to find candidate thresholds for a split
            sort_inputs_by = np.argsort(x_feature_index)
            x_feature_index_sorted, y_sorted = x_feature_index[sort_inputs_by], y[sort_inputs_by]

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
                tan93_score = ig / effective_cost
                if tan93_score > best_ig:
                    best_ig = tan93_score
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

    def _classification_cost_one(self, x_row, node, used_features: Optional[Set[str]]=None):
        if used_features is None:
            used_features : Set[str] = set()

        # Leaf reached, compute the total cost of getting to this leaf over all the used features
        if node.best_split is None:
            total_cost = 0.0
            for f in used_features:
                total_cost += self.feature_costs.get(f, 0.0)
            return total_cost

        feature_name = node.best_split.feature_name
        used_features.add(feature_name) # this is a set, feature name will only be added once during traversal
        if self.feature_cost_deps is not None:
            for deps in self.feature_cost_deps.get(feature_name, set()):
                used_features.add(deps)

        if x_row[node.best_split.feature_index] <= node.best_split.threshold:
            return self._classification_cost_one(x_row, node.left, used_features)
        else:
            return self._classification_cost_one(x_row, node.right, used_features)



