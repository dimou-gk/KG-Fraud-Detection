import sklearn
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
)
import time


def train_wth_centrality(clf_dictionary: dict, clfier: str, X: DataFrame, y: DataFrame):
    clf_pipeline = Pipeline(
        [
            ("feature_selection1", VarianceThreshold(threshold=0.0)),
            ("feature_selection2", SelectKBest(f_classif, k="all")),
            ("scaler", StandardScaler()),
            ("resampler1", SMOTE(sampling_strategy="minority", random_state=42)),
            (
                "resampler2",
                RandomUnderSampler(sampling_strategy="all", random_state=42),
            ),
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

    print(
        "Καλύτερα αποτελέσματα μέσω cross validation για {}:\n".format(
            clf_dictionary[clfier]["name"]
        )
    )
    print("{}\n".format(grid_search.best_params_))
    print("Best score: {:.3f}".format(grid_search.best_score_))
    return grid_search, skf


def test_with_centrality(
    clf_dictionary: dict, clfier: str, grid_search, skf, X: DataFrame, y: DataFrame
):
    time_needed = time.time()
    best_params = {k.split("__")[1]: v for k, v in grid_search.best_params_.items()}

    best_clf = clf_dictionary[clfier]["method"](**best_params)

    print("Αποτελεσματα {}:".format(clf_dictionary[clfier]["name"]))
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("Fold {}:".format(i + 1))

        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # X_temp_resampled, y_temp_resampled = knn_pipeline['resampler1'].fit_resample(X_train, y_train) #use resampler1
        # X_resampled, y_resampled = knn_pipeline['resampler2'].fit_resample(X_temp_resampled, y_temp_resampled) #use resampler2
        # best_clf.fit(X_resampled, y_resampled) #fit the classifier using the resampled data
        # y_pred = best_clf.predict(X_resampled)

        # non-resampled fit/predict
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

        conf_matr = confusion_matrix(y_test, y_pred)

        # print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
        print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
        print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
        print("F1: {:.2f}".format(f1_score(y_test, y_pred)))
        print("Confgusion matrix:")
        print(
            "\n".join(
                [" ".join(["{:5}".format(item) for item in row]) for row in conf_matr]
            )
        )
        # print(f"Elapsed time: {(time.time()-time_needed)/60} in seconds")
