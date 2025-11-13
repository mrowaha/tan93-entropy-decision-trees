import argparse
from typing import Optional

import pandas as pd

from tree import EntropyDecisionTree, Tan93EntropyDecisionTree
from utils.args import existing_file
from utils.datasets import evaluate

def main(args):
    train_df = pd.read_csv(args.train)
    x_train = train_df.drop(columns=["Performance_Label"])
    y_train = train_df["Performance_Label"].astype("category")

    test_df = pd.read_csv(args.test)
    x_test = test_df.drop(columns=["Performance_Label"])
    y_test = test_df["Performance_Label"].astype("category")

    model : Optional[EntropyDecisionTree] = None
    if args.cost is not None:
        print("model Tan93+Entropy DecisionTree")
        feature_costs = pd.read_csv(args.cost)
        feature_costs = feature_costs.set_index('Feature')['Cost'].to_dict()
        feature_cost_deps = {
            "ReliabilityIndex": {"ExperienceYears", "StressLevel"}
        }
        model = Tan93EntropyDecisionTree(feature_costs, feature_cost_deps=feature_cost_deps)
    else:
        print("model Entropy DecisionTree")
        model = EntropyDecisionTree()

    assert model is not None
    model.fit(x_train, y_train)

    if args.v:
        digraph = model.root.visualize()
        digraph.render(f"{model.name}_dt", format="png", cleanup=True)

    evaluate(model, "Custom Decision Tree", x_train, y_train, x_test, y_test)
    if isinstance(model, Tan93EntropyDecisionTree):
        classification_costs = model.classification_cost(x_train, y_train)
        print("classification costs on training data:", classification_costs)
        classification_costs = model.classification_cost(x_test, y_test)
        print("classification costs on test data:", classification_costs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", required=True, type=existing_file)
    parser.add_argument("-train", required=True, type=existing_file)
    parser.add_argument("-cost", type=existing_file)
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()
    main(args)