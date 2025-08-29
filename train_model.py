import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _read_split(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Простое чтение сплитов, подготовленных data_loader.py.
    Ожидается, что data_loader уже удалил служебные колонки и оставил нужные фичи.
    X = все колонки, кроме 'quality'; y = колонка 'quality'.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "quality" not in train_df.columns:
        raise ValueError("В train.csv должна быть колонка 'quality'.")
    if "quality" not in test_df.columns:
        raise ValueError("В test.csv должна быть колонка 'quality'.")

    X_train = train_df.drop(columns=["quality"])
    y_train = train_df["quality"].astype(int)
    X_test = test_df.drop(columns=["quality"])
    y_test = test_df["quality"].astype(int)


    X_test = X_test[X_train.columns]
    return X_train, X_test, y_train, y_test


def _experiments(seed: int) -> Dict[str, Pipeline]:
    logreg = Pipeline([
        ("scaler", StandardScaler()),  # почему: LR чувствительна к масштабу
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])

    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    rf_tuned = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1,
    )

    return {
        "logreg": logreg,
        "rf_base": Pipeline([("clf", rf_base)]),
        "rf_tuned": Pipeline([("clf", rf_tuned)]),
    }


def _train_one(name: str, pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro"))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1m)
    mlflow.log_metric("train_time_sec", train_time)
    return {"accuracy": acc, "f1_macro": f1m, "train_time_sec": train_time}


def ensure_experiment(name: str) -> None:
    """Создаёт/восстанавливает эксперимент, если он удалён."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        mlflow.set_experiment(name)
        return
    if getattr(exp, "lifecycle_stage", "active") == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Wine Quality models (MLflow, семинар-стиль)")
    p.add_argument("--train", type=str, default="data/processed/train.csv")
    p.add_argument("--test", type=str, default="data/processed/test.csv")
    p.add_argument("--experiment-name", type=str, default="wine-quality")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    X_train, X_test, y_train, y_test = _read_split(args.train, args.test)
    class_counts = y_train.value_counts().to_dict()

    ensure_experiment(args.experiment_name)
    exps = _experiments(args.seed)

    best_name = None
    best_acc = -1.0
    best_model = None

    for name, pipe in exps.items():
        run_name = f"{name}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(run_name=run_name):
            # Теги и параметры датасета
            mlflow.set_tag("algo", name)
            mlflow.set_tag("run_ts", run_name.rsplit("_", 1)[-1])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train", X_train.shape[0])
            mlflow.log_param("n_test", X_test.shape[0])
            mlflow.log_param("class_counts", json.dumps(class_counts))
            mlflow.log_param("model", name)
            if hasattr(pipe, "get_params"):
                for k, v in pipe.get_params(deep=False).items():
                    if isinstance(v, (int, float, str, bool)):
                        mlflow.log_param(k, v)

            metrics = _train_one(name, pipe, X_train, y_train, X_test, y_test)

            input_example = X_train.head(5)
            try:
                signature = infer_signature(input_example, pipe.predict(input_example))
            except Exception:
                signature = None
            mlflow.sklearn.log_model(pipe, name="model", signature=signature, input_example=input_example)

            out_dir = Path("model_weights"); out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(pipe, f)

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_name = name
                best_model = pipe

    if best_model is not None:
        best_path = Path("model_weights") / "best_model.pkl"
        with open(best_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Best: {best_name} | accuracy={best_acc:.4f} | saved: {best_path}")
    else:
        print("No model was trained.")


if __name__ == "__main__":
    main()
