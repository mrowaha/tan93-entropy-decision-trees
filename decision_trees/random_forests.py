import argparse
from typing import Optional

import pandas as pd

from tree import EntropyRandomForest
from utils.args import existing_file
from utils.datasets import evaluate

def main(args):
    train_df = pd.read_csv(args.train)
    x_train = train_df.drop(columns=["Performance_Label"])
    y_train = train_df["Performance_Label"].astype("category")

    test_df = pd.read_csv(args.test)
    x_test = test_df.drop(columns=["Performance_Label"])
    y_test = test_df["Performance_Label"].astype("category")

    max_features = args.max_features
    balance_classes = args.balance
    random_split = args.random

    model = EntropyRandomForest(
        max_features=6 if max_features else None,
        balance_classes=balance_classes,
        random_split=random_split
    )

    model.fit(x_train, y_train)
    evaluate(model, f"Random Forest Configuration: max features (6): {max_features}, balance classes: {balance_classes}, random split: {random_split}", x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", required=True, type=existing_file)
    parser.add_argument("-train", required=True, type=existing_file)
    parser.add_argument("-max-features", action="store_true")
    parser.add_argument("-balance", action="store_true")
    parser.add_argument("-random", action="store_true")
    args = parser.parse_args()
    main(args)