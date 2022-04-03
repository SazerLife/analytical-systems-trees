import warnings

warnings.filterwarnings("ignore")
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from data_process import get_grades_csv

train_sizes = [round(i / 100, 1) for i in range(60, 91, 10)]


def run():
    X, y = get_grades_csv()
    f1_scores = list()
    for tarin_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=tarin_size, random_state=0
        )
        clf = DecisionTreeClassifier(random_state=0, criterion="gini")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted", labels=np.unique(y_pred))
        f1_scores.append(f1)

        fig = plt.figure(figsize=(25, 20))
        plot_tree(clf)
        plt.savefig(f"tree_plots/{tarin_size}.png")
        plt.close("all")

    print(f1_scores)
    plt.plot(train_sizes, f1_scores)
    plt.savefig("f1_from_train_sizes.png")


if __name__ == "__main__":
    run()
