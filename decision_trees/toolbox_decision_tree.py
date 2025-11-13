import argparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from utils.args import existing_file
from utils.datasets import label_counts_as_percentage, evaluate

RANDOM_STATE = 42
params = {
    'clf__max_depth': range(1, 10, 1),
    'clf__min_samples_split': range(2, 20, 2),
    'clf__criterion': ["entropy", "gini"]
}

def save_decision_tree(clf, feature_names, class_names, file_name):
    plt.figure(figsize=(24, 20))
    plot_tree(clf, feature_names=feature_names,
              class_names=class_names,
              filled=True, rounded=True, fontsize=7)
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close()

def run_fine_tuned_pipeline(X_train, y_train, X_test, y_test):
    global params, RANDOM_STATE
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_estimator = gs.best_estimator_
    fined_tuned_best_params = gs.best_params_
    print(f"Fine tuned best params: {fined_tuned_best_params}")
    evaluate(best_estimator, "Fine tuned (pre-pruned) without normalization", X_train, y_train, X_test, y_test)
    return best_estimator

def run_no_pruned_pipeline(X_train, y_train, X_test, y_test):
    global RANDOM_STATE
    no_prune_pipeline = Pipeline([
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE,
                                       criterion="entropy", max_depth=None, min_samples_split=2)),
    ])
    no_prune_pipeline.fit(X_train, y_train)
    evaluate(no_prune_pipeline, "Without pruning", X_train, y_train, X_test, y_test)
    return no_prune_pipeline

def run_normalized_pipeline(X_train, y_train, X_test, y_test):
    global params, RANDOM_STATE
    binary_cols = ['LeadershipRole']
    continuous_cols = [c for c in X_train.columns if c not in binary_cols]
    preprocess = ColumnTransformer(
        transformers=[
            ("scale_continuous", StandardScaler(), continuous_cols),
            ("pass_binary", "passthrough", binary_cols),
        ]
    )
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_estimator = gs.best_estimator_
    best_params = gs.best_params_
    print(f"Best params: {best_params}")
    evaluate(best_estimator, "Fine tuned with normalization", X_train, y_train, X_test, y_test)
    return best_estimator

def run_undersampled_pipeline(X_train, y_train, X_test, y_test):
    global params, RANDOM_STATE
    rus = RandomUnderSampler(random_state=RANDOM_STATE)  # sampling_strategy='auto' (default) == match minority count
    _, y_under_all = rus.fit_resample(X_train, y_train)

    print("After underSampling:")
    print(label_counts_as_percentage(y_under_all))
    pipeline = ImbPipeline([
        ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipeline, params, cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best_under = gs.best_estimator_
    print("Best (undersampled):", gs.best_params_)
    evaluate(best_under, "Fine Tuned with undersampling", X_train, y_train, X_test, y_test)
    return best_under

def main(args):
    train_df = pd.read_csv(args.train)
    X_train = train_df.drop(columns=["Performance_Label"])
    y_train = train_df["Performance_Label"].astype("category")

    print("Original training distribution (y_train):")
    print(label_counts_as_percentage(y_train))

    test_df = pd.read_csv(args.test)
    X_test = test_df.drop(columns=["Performance_Label"])
    y_test = test_df["Performance_Label"].astype("category")

    fine_tuned_estimator = run_fine_tuned_pipeline(X_train, y_train, X_test, y_test)
    run_no_pruned_pipeline(X_train, y_train, X_test, y_test)
    run_normalized_pipeline(X_train, y_train, X_test, y_test)
    run_undersampled_pipeline(X_train, y_train, X_test, y_test)

    feature_names, class_names = X_train.columns.tolist(), y_train.cat.categories.astype(str).tolist()
    save_decision_tree(
        fine_tuned_estimator.named_steps["clf"],
        feature_names,
        class_names,
        "fine_tuned_decision_tree.png"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", required=True, type=existing_file)
    parser.add_argument("-train", required=True, type=existing_file)
    args = parser.parse_args()
    main(args)