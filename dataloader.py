from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLS: List[str] = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Прямое чтение CSV. Оставляем простым — важно для воспроизводимости."""
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Мини-очистка: убираем служебные колонки и нормализуем имя цели."""
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    for c in ("Id", "id"):
        if c in df.columns:
            df = df.drop(columns=[c])
    if "quality" not in df.columns and "Quality" in df.columns:
        df = df.rename(columns={"Quality": "quality"})
    if "quality" not in df.columns:
        raise ValueError("Не найден столбец 'quality' в CSV")
    return df


def select_features(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Проверяем наличие столбцов и возвращаем X ровно из feature_cols."""
    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют признаки: {missing}")
    return df[feature_cols].copy()


def make_target(quality: pd.Series, task: str, threshold: int) -> pd.Series:
    """Единая логика формирования цели. Бинаризация по порогу — дефолт."""
    quality = quality.astype(int)
    if task == "binary":
        y = (quality >= threshold).astype(int)
    elif task == "multiclass":
        y = quality
    else:
        raise ValueError("task должен быть 'binary' или 'multiclass'")
    return y.rename("quality")


def split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    seed: int,
    task: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if task in {"binary", "multiclass"} else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def save_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    out_dir: str | Path,
) -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.csv"
    test_path = out / "test.csv"
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)
    return train_path, test_path


def load_and_split(
    *,
    csv_path: str,
    task: str = "binary",
    threshold: int = 6,
    feature_cols: Iterable[str] = FEATURE_COLS,
    test_size: float = 0.2,
    seed: int = 42,
    save_to: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Высокоуровневая точка входа: читает CSV, чистит, строит X/y, сплитит, опционально сохраняет."""
    df = load_dataset(csv_path)
    df = preprocess(df)
    X = select_features(df, feature_cols)
    y = make_target(df["quality"], task=task, threshold=threshold)
    X_train, X_test, y_train, y_test = split(X, y, test_size=test_size, seed=seed, task=task)
    if save_to is not None:
        save_splits(X_train, X_test, y_train, y_test, out_dir=save_to)
    return X_train, X_test, y_train, y_test


# --------------------- CLI ---------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data loader for Wine Quality (с функциями)")
    p.add_argument("--in", dest="csv_path", type=str, required=True, help="Путь к исходному CSV (WineQT.csv)")
    p.add_argument("--out-dir", type=str, default="data/processed", help="Куда писать train/test")
    p.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--threshold", type=int, default=6, help="quality >= threshold -> класс 1 (для binary)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    X_train, X_test, y_train, y_test = load_and_split(
        csv_path=args.csv_path,
        task=args.task,
        threshold=args.threshold,
        feature_cols=FEATURE_COLS,
        test_size=args.test_size,
        seed=args.seed,
        save_to=args.out_dir,
    )
    print("train:", X_train.shape, ", test:", X_test.shape)


if __name__ == "__main__":
    main()
