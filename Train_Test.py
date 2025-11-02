import sklearn
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

import matplotlib.pyplot as plt


def train_wth_centrality(clf_dictionary: dict, clfier: str, X: DataFrame, y: DataFrame):
    clf_pipeline = Pipeline(
        [
            ("feature_selection1", VarianceThreshold(threshold=0.0)),
            ("feature_selection2", SelectKBest(f_classif, k="all")),
            ("scaler", StandardScaler()),
            # ("resampler1", SMOTE(sampling_strategy="minority", random_state=42)),
            # (
            #     "resampler2",
            #     RandomUnderSampler(sampling_strategy="all", random_state=42),
            # ),
            ("clf", clf_dictionary[clfier]["method"]()),
        ],
        # verbose=True
    )

    params_grid = clf_dictionary[clfier]["params_grid"]

    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    grid_search = sklearn.model_selection.GridSearchCV(
        clf_pipeline,
        params_grid,
        scoring=["precision", "recall"],
        refit="precision",
        cv=skf,
    )

    grid_search.fit(X, y)

    # Best features
    # mask = grid_search.best_estimator_.named_steps["feature_selection2"].get_support()
    # new_features = X.columns[mask]
    # print(f"Best features selected by the feature selector: {new_features}")

    print(
        "Καλύτερα αποτελέσματα μέσω cross validation για {}:\n".format(
            clf_dictionary[clfier]["name"]
        )
    )
    print("{}\n".format(grid_search.best_params_))
    print("Best score: {:.3f}".format(grid_search.best_score_))
    return grid_search, skf


def test_with_centrality(
    clf_dictionary: dict,
    clfier: str,
    grid_search,
    skf,
    X: DataFrame,
    y: DataFrame,
    centrality_name: str = "default",
):
    best_params = {k.split("__")[1]: v for k, v in grid_search.best_params_.items()}

    best_clf = clf_dictionary[clfier]["method"](**best_params)

    prec = 0
    recall = 0
    f1 = 0
    conf_matrix_list_of_arrays = []

    print("Αποτελεσματα {}:".format(clf_dictionary[clfier]["name"]))

    roc_curves = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("Fold {}:".format(i + 1))

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # non-resampled fit/predict
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

        conf_matr = confusion_matrix(y_test, y_pred)

        # Store metrics to average them
        prec += precision_score(y_test, y_pred)
        recall += recall_score(y_test, y_pred)
        f1 += f1_score(y_test, y_pred)
        conf_matrix_list_of_arrays.append(conf_matr)

        # Probability predictions for ROC
        if hasattr(best_clf, "predict_proba"):
            y_score = best_clf.predict_proba(X_test)[:, 1]
        else:
            y_score = best_clf.decision_function(X_test)

        # Save one ROC curve (or average)
        # fpr, tpr, _ = roc_curve(y_test, y_score)
        # roc_data = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        # path = (
        #     "roc_data_"
        #     + clf_dictionary[clfier]["name"]
        #     + "_quadruple_Unweigh_Degree_Unweigh_Eigen_Betwe_Close_centrality.csv"
        # )
        # roc_data.to_csv(path, index=False)

        # Compute ROC for this fold
        fpr, tpr, _ = roc_curve(y_test, y_score)
        fold_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "fold": i + 1})
        roc_curves.append(fold_df)

        print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
        print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
        print("F1: {:.2f}".format(f1_score(y_test, y_pred)))
        print("Confusion matrix:")
        print(
            "\n".join(
                [" ".join(["{:5}".format(item) for item in row]) for row in conf_matr]
            )
        )
    print(f"Average Precision is: {prec/5}")
    print(f"Average Recall is: {recall/5}")
    print(f"Average F1 is: {f1/5}")
    print(f"Average Confusion Matrix is: {np.mean(conf_matrix_list_of_arrays, axis=0)}")

    # -------------------------------------#
    # Combine ROC data from all folds
    roc_data = pd.concat(roc_curves, ignore_index=True)

    # --- dynamic naming for ROC CSV ---
    import os
    from datetime import datetime

    os.makedirs("roc_results", exist_ok=True)  # keep them organized

    clf_name = clf_dictionary[clfier]["name"].replace(" ", "_")

    filename = f"roc_results/ROC_{clf_name}_{centrality_name}.csv"

    roc_data.to_csv(filename, index=False)

    # -------------------------------------#
    # Feature Importance (Only for Logistic Regression)
    # if clfier == "clf2":
    #     features = X.columns
    #     coefficients = best_clf.coef_[0]
    #     # print(coefficients)
    #     # print(type(coefficients))

    #     plt.barh(features, coefficients)
    #     plt.title("Coefficients of Linear Regression")
    #     plt.xlabel("Features")
    #     plt.ylabel("Coefficients")
    #     plt.show()
