import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from Train_Test import train_wth_centrality, test_with_centrality
from datetime import datetime
import time


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
        # "clf3": {
        #     "name": "SVM",
        #     "method": SVC,
        #     "params_grid": {
        #         "clf__C": [1],  # [1, 10]
        #         "clf__kernel": ["linear"],
        #         #'clf__degree': [2],
        #         #'clf__gamma': ['scale']
        #     },
        # },
        "clf3": {
            "name": "LinearSVC",
            "method": LinearSVC,
            "params_grid": {
                "clf__C": [1],  # [1, 10]
                "clf__penalty": ["l1"],
                "clf__loss": ["squared_hinge"],
                "clf__dual": [False],
                "clf__max_iter": [1500],
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
                    LogisticRegression(max_iter=1000),
                ],
                "clf__n_estimators": [3],
                "clf__n_jobs": [2],
            },
        },
        # "clf8": {
        #     "name": "Stacking",
        #     "method": lambda: StackingClassifier(
        #         estimators=[
        #             ("xgb", XGBClassifier()),
        #             ("knn", KNeighborsClassifier()),
        #             ("logreg", LogisticRegression(max_iter=1000)),
        #         ],
        #         final_estimator=LogisticRegression(max_iter=1000),
        #     ),
        #     "params_grid": {
        #         # "clf__estimators": [
        #         #     XGBClassifier(),
        #         #     KNeighborsClassifier(),
        #         #     LogisticRegression(max_iter=1000),
        #         # ],
        #     },
        # },
    }

    uri = "bolt://localhost:7687/kg1"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))

    # uri = "bolt://localhost:7687/kgorigminusdestdiffweights2"
    # driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))

    # Baseline
    query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
                MATCH (t)-[at:AT]->(tm:Time)
                RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
                        t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
                        with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
                        v.name AS dest_name,
                        tm.hour AS time_hour
    """

    # Single (Hits) Centrality
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub,
    #                     v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub,
    #                     tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub
    # """

    # Double (Degree & Hits) Centralities
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.degree AS org_degree,  u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub,
    #                     v.degree AS dest_degree,  v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub,
    #                     tm.degree AS time_degree,  tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub
    # """

    # Triple (Hits & Eigenvector & Pagerank) Centralities
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub, u.eigenvector_unweighted as org_eigenvector_unweighted, u.pagerank as org_pagerank,
    #                     v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub, v.eigenvector_unweighted as dest_eigenvector_unweighted, v.pagerank as dest_pagerank,
    #                     tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub, tm.eigenvector_unweighted as time_eigenvector_unweighted, tm.pagerank as time_pagerank
    # """

    # Quadruple (Degree & Eigenvector & Betweenness & Closeness) Centralities
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.degree AS org_degree, u.eigenvector_unweighted as org_eigenvector_unweighted, u.betweenness as org_betweenness, u.closeness as org_closeness,
    #                     v.degree AS dest_degree, v.eigenvector_unweighted as dest_eigenvector_unweighted, v.betweenness as dest_betweenness, v.closeness as dest_closeness,
    #                     tm.degree AS time_degree, tm.eigenvector_unweighted as time_eigenvector_unweighted, tm.betweenness as time_betweenness, tm.closeness as time_closeness
    # """

    # Removal of features
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN
    #                     t.transType AS trans_type, t.isFraud AS isFraud,
    #                     tm.hour AS time_hour,
    #                     u.degree AS org_degree,  u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub,
    #                     v.degree AS dest_degree,  v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub,
    #                     tm.degree AS time_degree,  tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub
    # """

    # ------------------------------------------------------------------------------------------------#
    # kgorigminusdest Database queries
    # Single (Hits) Centrality
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub,
    #                     v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub,
    #                     tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub
    # """

    # Double (Degree & Hits) Centralities
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.degree_weighted AS org_degree,  u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub,
    #                     v.degree_weighted AS dest_degree,  v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub,
    #                     tm.degree_weighted AS time_degree, tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub
    # """

    # 3+ Centralities
    # query = """ MATCH (u:User)-[made:MADE_A]->(t:Transaction)-[with:WITH_]->(v:User)
    #             MATCH (t)-[at:AT]->(tm:Time)
    #             RETURN u.name AS org_name, made.oldbalanceOrg as oldbalanceOrg, made.newbalanceOrg as newbalanceOrg,
    #                     t.amount AS amount, t.transType AS trans_type, t.isFraud AS isFraud,
    #                     with.oldbalanceDest as oldbalanceDest, with.newbalanceDest as newbalanceDest,
    #                     v.name AS dest_name,
    #                     tm.hour AS time_hour,
    #                     u.hitsAuth as org_hitsAuth, u.hitsHub as org_hitsHub, u.eigenvector_weighted as org_eigenvector_weighted,
    #                     v.hitsAuth as dest_hitsAuth, v.hitsHub as dest_hitsHub, v.eigenvector_weighted as dest_eigenvector_weighted,
    #                     tm.hitsAuth as time_hitsAuth, tm.hitsHub as time_hitsHub, tm.eigenvector_weighted as time_eigenvector_weighted
    # """

    with driver.session(database="kg1") as session:
        result = session.run(query)
        print(
            f"I have reached this checkpoint (after query run) at time: {datetime.now()}"
        )
        df = result.to_df()
        print(
            f"I have reached this checkpoint (after df tranform) at time: {datetime.now()}"
        )

    # Sampling
    # df = df.sample(frac=0.75)

    encoder = {}
    for i in df.select_dtypes("object").columns:
        encoder[i] = LabelEncoder()
        df[i] = encoder[i].fit_transform(df[i])
    print(f"I have reached this checkpoint (after encoder) at time: {datetime.now()}")

    # Fill NaN or our weighted eigenvector in dest nodes on kgorigminusdest database
    df.fillna(value=0, inplace=True)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # Try a single Algorithm
    # Change the "clf" string inside the train_with_centrality & test_with_centrality
    # function parameters according to which algorithm you want to execute
    # (based on the classifiers_dict)
    time_needed = time.time()
    grid_search_results, stratifks = train_wth_centrality(
        classifiers_dict, "clf7", X, y
    )
    test_with_centrality(classifiers_dict, "clf7", grid_search_results, stratifks, X, y)
    print(f"Elapsed time: {(time.time()-time_needed)/60} minutes")

    # Try all the Classification Algorithms in a loop
    # for clf_key in classifiers_dict:
    #     time_needed = time.time()

    #     try:
    #         # Train the model with cross-validation
    #         grid_search_results, skf = train_wth_centrality(
    #             classifiers_dict, clf_key, X, y
    #         )

    #         # Test the model
    #         test_with_centrality(
    #             classifiers_dict, clf_key, grid_search_results, skf, X, y
    #         )

    #         print(f"Elapsed time: {(time.time()-time_needed)/60} minutes")

    #     except Exception as e:
    #         print(f"An error occurred with classifier {clf_key}: {e}")


if __name__ == "__main__":
    main()
