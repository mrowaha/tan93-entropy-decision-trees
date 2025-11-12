import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
