import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from Train_Test import train_wth_centrality, test_with_centrality


def main():
    classifiers_dict = {
        "clf1": {
            "name": "K-Nearest Neighbor",
            "method": KNeighborsClassifier,
            "params_grid": {"clf__n_neighbors": [4], "clf__weights": ["uniform"]},
        },
        "clf2": {
            "name": "Logistic Regression",
            "method": LogisticRegression,
            "params_grid": {
                "clf__C": np.logspace(-6, -5, 10),
                "clf__max_iter": [1000],
            },
        },
        "clf3": {
            "name": "SVM",
            "method": SVC,
            "params_grid": {
                "clf__C": [1],  # [1, 10]
                "clf__kernel": ["linear"]
                #'clf__degree': [2],
                #'clf__gamma': ['scale']
            },
        },
        "clf4": {
            "name": "Naive_Bayes",
            "method": GaussianNB,
            "params_grid": {
                "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
                "clf__priors": [None, [0.4, 0.6], [0.3, 0.7]],
            },
        },
        "clf5": {
            "name": "Decision_Tree",
            "method": DecisionTreeClassifier,
            "params_grid": {
                "clf__criterion": ["gini", "entropy", "log_loss"],
                "clf__max_leaf_nodes": [4, 5, 6],
            },
        },
        "clf6": {
            "name": "Random_forest",
            "method": RandomForestClassifier,
            "params_grid": {
                "clf__n_estimators": [50, 100, 150],
                "clf__criterion": ["gini", "entropy"],
            },
        },
        "clf7": {
            "name": "XGBoost",
            "method": XGBClassifier,
            "params_grid": {"clf__n_estimators": range(50, 150, 50)},
        },
        "clf8": {
            "name": "Bagging",
            "method": BaggingClassifier,
            "params_grid": {
                "clf__estimator": [
                    XGBClassifier(),
                    KNeighborsClassifier(),
                    LogisticRegression(),
                ],
                "clf__n_estimators": [3],
                "clf__n_jobs": [-1],
            },
        },
    }

    uri = "bolt://localhost:7687/kg1"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))

    query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
                MATCH (t)-[at:AT]->(tm:Time)
                RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
                        t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
                        with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
                        v.name AS dest_name,
                        tm.hour AS time_hour,
                        u.degree AS org_degree,  u.eigenvector_unweighted as org_eigenvector_unweighted,
                        v.degree AS dest_degree,  v.eigenvector_unweighted as dest_eigenvector_unweighted,
                        tm.degree AS time_degree,  tm.eigenvector_unweighted as time_eigenvector_unweighted
    """

    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.degree AS org_degree,
    #                     t.degree as trans_degree,
    #                     v.degree AS dest_degree,
    #                     tm.degree AS time_degree
    # """

    with driver.session(database="kg1") as session:
        result = session.run(query)
        df = result.to_df()

        encoder = {}
        for i in df.select_dtypes("object").columns:
            encoder[i] = LabelEncoder()
            df[i] = encoder[i].fit_transform(df[i])
        # df.info(verbose=True)
        # path = 'D:\General\Microsoft VS Code projects\Python\Thesis\encoded.csv'
        # df.to_csv(path, index = False)

        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]

        grid_search_results, stratifks = train_wth_centrality(
            classifiers_dict, "clf6", X, y
        )
        test_with_centrality(
            classifiers_dict, "clf6", grid_search_results, stratifks, X, y
        )


if __name__ == "__main__":
    main()
