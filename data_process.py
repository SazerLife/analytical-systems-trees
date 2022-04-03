from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parents[0]


def get_adult_dataset() -> Tuple[np.array, np.array]:
    df1 = pd.read_csv("data/adult.data", header=None)
    df2 = pd.read_csv("data/adult.test", header=None)

    return pd.concat([df1, df2])


def get_grades_csv() -> Tuple[np.array, np.array]:
    df = pd.read_csv(PROJECT_DIR / "data/grades/grades.csv")
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int8)
    CATEGORICAL_X = np.asarray(df[df.columns[0:2]].values)
    X = np.asarray(df[df.columns[2:8]].values)
    y = np.asarray(df[df.columns[8]].to_list())

    BIN_X = one_hot_encoder.fit_transform(CATEGORICAL_X)
    X = np.concatenate((BIN_X, X), axis=1)  # .reshape(10, -1)

    return X, y
