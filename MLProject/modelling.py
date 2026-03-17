import mlflow # type: ignore
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import argparse
import numpy as np
from manual_logging import log_dataset, log_metric, log_model, log_artifact


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Handle Argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=str, default="None")
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--train", type=str, default="CropRecommendation_preprocessing/train_preprocessed.csv")
    parser.add_argument("--test", type=str, default="CropRecommendation_preprocessing/test_preprocessed.csv")

    args = parser.parse_args()

    max_depth = None if args.max_depth == "None" else int(args.max_depth)
    n_estimators = args.n_estimators
    min_samples_leaf = args.min_samples_leaf

    DATA_TRAIN_PATH = args.train
    DATA_TEST_PATH = args.test

    data_train = pd.read_csv(DATA_TRAIN_PATH)
    data_test = pd.read_csv(DATA_TEST_PATH)

    X_train = data_train.drop("label", axis=1)
    y_train = data_train["label"]
    X_test = data_test.drop("label", axis=1)
    y_test = data_test["label"]

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=f"model_{n_estimators}_{max_depth}_{min_samples_leaf}"):

        log_dataset(data_train=data_train, data_train_path=DATA_TRAIN_PATH, data_test=data_test, data_test_path=DATA_TEST_PATH)

        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)
        y_pred_proba = rfc.predict_proba(X_test)

        mlflow.log_params(rfc.get_params())
        log_metric(model=rfc, X_test=X_test, y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
        log_model(model=rfc, input_example=input_example)
        log_artifact(model=rfc, X_train=X_train, y_test=y_test, y_pred=y_pred)