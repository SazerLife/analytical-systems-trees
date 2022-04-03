import warnings

warnings.filterwarnings("ignore")
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder

from data_process import get_grades_csv

iterations = [i for i in range(50, 101, 10)]


def run():
    X, y = get_grades_csv()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    f1_scores = list()
    for iteration in iterations:
        clf = CatBoostClassifier(
            auto_class_weights="SqrtBalanced",
            loss_function="MultiClass",
            iterations=iteration,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred))
        f1_scores.append(f1)

    print(f1_scores)
    plt.plot(iterations, f1_scores)
    plt.savefig("f1_from_iterations.png")


if __name__ == "__main__":
    run()
