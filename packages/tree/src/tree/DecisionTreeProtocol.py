from typing import Protocol, Self

import pandas as pd

class DecisionTreeProtocol(Protocol):
    def fit(self: Self, x_in: pd.DataFrame, y: pd.Series):
        ...

    def predict(self: Self, x_in: pd.DataFrame) -> pd.Series:
        ...