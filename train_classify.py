import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_process import get_grades_csv


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="grades")
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--train-size", type=float, default=0.8)

    return parser


def classify_data(X: np.array, y: np.array, criterion: str, tarin_size: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=tarin_size, random_state=0
    )
    clf = DecisionTreeClassifier(random_state=0, criterion=criterion)
    clf.fit(X_train, y_train)
    return y_test, clf.predict(X_test)


def main(dataset: str, criterion: str, train_size: float):
    if dataset == "":
        pass
    elif dataset == "grades":
        X, y = get_grades_csv()
    else:
        raise RuntimeError("")

    y_test, y_pred = classify_data(X, y, criterion, train_size)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    main(args.dataset, args.criterion, args.train_size)
