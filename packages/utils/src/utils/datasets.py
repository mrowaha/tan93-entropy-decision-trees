from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

def label_counts_as_percentage(y) -> pd.DataFrame:
    classes = list(y.cat.categories)
    counts = y.value_counts().reindex(classes, fill_value=0)
    perc = (counts / counts.sum() * 100.0).round(2)
    return pd.DataFrame({"count": counts.astype(int), "percent": perc}, index=classes)

def per_class_accuracy(y_true, y_pred):
    labels = list(y_true.cat.categories)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1, keepdims=True)
    per_cls = np.divide(np.diag(cm), support.ravel(),
                        out=np.zeros_like(support.ravel(), dtype=float),
                        where=support.ravel()!=0)
    return pd.Series(per_cls, index=labels, name="per_class_accuracy")

def entropy(y):
    """Compute entropy of label array y."""
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def get_information_gain_by_n(n) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    def calculate_information_gain(y, left_mask, right_mask):
        nonlocal n
        H = entropy(y)
        y_left, y_right = y[left_mask], y[right_mask]
        H_left = entropy(y_left)
        H_right = entropy(y_right)

        H_split = (len(y_left) / n) * H_left + \
                  (len(y_right) / n) * H_right
        return H - H_split
    return calculate_information_gain

def get_class_counts(n_classes: int, y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    class_counts = np.zeros(n_classes, dtype=int)
    class_counts[values] = counts
    return class_counts

def evaluate(model, name, X_tr, y_tr, X_te, y_te):
    print(f"\n=== {name} ===")
    yhat_tr = model.predict(X_tr); yhat_te = model.predict(X_te)
    print(f"Accuracy (train): {accuracy_score(y_tr, yhat_tr):.4f}")
    print(f"Accuracy (test) : {accuracy_score(y_te, yhat_te):.4f}")
    print("\nPer-class accuracy (train):"); print(per_class_accuracy(y_tr, yhat_tr))
    print("\nPer-class accuracy (test):");  print(per_class_accuracy(y_te, yhat_te))
    ConfusionMatrixDisplay.from_predictions(y_tr, yhat_tr)
    plt.title(f"{name} — Train CM"); plt.show()
    ConfusionMatrixDisplay.from_predictions(y_te, yhat_te)
    plt.title(f"{name} — Test CM");  plt.show()
