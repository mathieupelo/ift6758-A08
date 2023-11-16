import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

from scripts import feature_engineering
def generateGraphsGridSearch():
    data = pd.read_csv('../data/derivatives/train_data.csv')
    x2, y2 = feature_engineering.preprocessing(data, "is_goal")
    X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)

    xgboost_classifier = xgb.XGBClassifier()
    xgboost_classifier.fit(X2_train, y2_train)

    param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }

    # Create the GridSearchCV object with cross-validation (e.g., 5-fold cross-validation)
    grid_search = GridSearchCV(
        estimator=xgboost_classifier,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5
    )


    print("Training with the Grid Search")
    # We train with the Grid Search
    grid_search.fit(X2_train, y2_train)

    import matplotlib.pyplot as plt

    # ... (your existing code)

    # Print the best parameters and the corresponding ROC AUC score
    print("Best Parameters: ", grid_search.best_params_)
    print("Best ROC AUC Score: {:.4f}".format(grid_search.best_score_))

    # Extract the results from the GridSearchCV
    results = grid_search.cv_results_

    # Extract the values of n_estimators and their corresponding mean test scores (ROC AUC)
    n_estimators_values = results['param_n_estimators'].data.astype(int)
    mean_test_scores = results['mean_test_score']

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_values, mean_test_scores, marker='o', linestyle='-')
    plt.title('Effect of n_estimators on ROC AUC Score')
    plt.xlabel('n_estimators')
    plt.ylabel('ROC AUC Score')
    plt.grid(True)
    plt.show()

    max_depth_values = results['param_max_depth'].data.astype(int)
    mean_test_scores = results['mean_test_score']

    # Plot the results for max_depth
    plt.figure(figsize=(10, 6))
    plt.plot(max_depth_values, mean_test_scores, marker='o', linestyle='-')
    plt.title('Effect of max_depth on ROC AUC Score')
    plt.xlabel('max_depth')
    plt.ylabel('ROC AUC Score')
    plt.grid(True)
    plt.show()

    # Extract the results from the GridSearchCV for learning_rate
    results = grid_search.cv_results_

    # Extract the values of learning_rate and their corresponding mean test scores (ROC AUC)
    learning_rate_values = results['param_learning_rate'].data.astype(float)
    mean_test_scores = results['mean_test_score']

    # Plot the results for learning_rate
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate_values, mean_test_scores, marker='o', linestyle='-')
    plt.title('Effect of learning_rate on ROC AUC Score')
    plt.xlabel('learning_rate')
    plt.ylabel('ROC AUC Score')
    plt.xscale('log')  # Use log scale for better visualization if learning_rate values vary widely
    plt.grid(True)
    plt.show()

generateGraphsGridSearch()

