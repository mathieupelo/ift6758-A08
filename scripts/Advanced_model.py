import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from comet_ml import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scripts.Plots import *
import feature_engineering

def xgboost(isGridSearch):
    def gridSearch(model, X2_train, y2_train):
        # Define the hyperparameters to tune and their possible values
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        # Create the GridSearchCV object with cross-validation (e.g., 5-fold cross-validation)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=5
        )

        print("Training with the Grid Search")
        # We train with the Grid Search
        grid_search.fit(X2_train, y2_train)

        # We output the best parameters that were given
        print("Output the best hyperparams")
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters: {best_params}")
        return grid_search

    def featureSelection(best_xgboost_classifier, y_test, X_train, y_train, X_test):
        # XGBoost has a L1 regularization term that we can use to identify features that are assigned the 0 weight
        xgboost_classifier = xgb.XGBClassifier(learning_rate=best_xgboost_classifier.learning_rate, n_estimators=best_xgboost_classifier.n_estimators, max_depth=best_xgboost_classifier.max_depth, reg_alpha=1)  # Add reg_alpha for L1 regularization
        xgboost_classifier.fit(X_train, y_train)

        importance_scores = xgboost_classifier.feature_importances_
        feature_names = X_train.columns
        feature_importance = list(zip(feature_names, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        threshold = np.percentile(importance_scores, 60)

        top_features = []
        for feature, importance in feature_importance:
            if importance > threshold:
                top_features.append(feature)

        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        xgboost_classifier.fit(X_train_top, y_train)

        # We evaluate once again
        y_probs = xgboost_classifier.predict_proba(X_test_top)

        y_test = pd.Series(y_test)
        CLFS = [[[xgboost_classifier], X_test_top, 'XGBoost Feature Selection']]
        Ys = [["XGBoost Feature Selection", y_probs[:, 1], "blue", True]]

        ROC_plot(y_test, Ys)
        Centiles_plot(y_test, Ys)
        cumulative_centiles_plot(y_test, Ys)
        calibrate_display(CLFS, y_test)


        return xgboost_classifier


    # 1 .
    data = pd.read_csv('../data/derivatives/train_data.csv')
    experiment = Experiment(
        api_key="Bgx9192SVK3nzJNLQcV5nneQS",
        project_name="milestone-2",
        workspace="me-pic"
    )
    experiment.set_name('XGBoost')

    X = data[['shot_distance', 'shot_angle']]
    y = data['is_goal']

    # On split 80 / 20 les donnees
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xgboost_classifier = xgb.XGBClassifier()
    xgboost_classifier.fit(X_train, y_train)

    #y_pred_proba = xgboost_classifier.predict_proba(X_test)[:, 1]

    y_probs = xgboost_classifier.predict_proba(X_test)
    y_test = pd.Series(y_test)
    CLFS = [[[xgboost_classifier], X_test, 'XGBoost']]
    Ys = [["XGBoost", y_probs[:, 1], "blue", True]]

    ROC_plot(y_test, Ys)
    Centiles_plot(y_test, Ys)
    cumulative_centiles_plot(y_test, Ys)
    calibrate_display(CLFS, y_test)

    # 2.
    x2, y2 = feature_engineering.preprocessing(data, "is_goal")
    X2_train, X2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2)
    xgboost_classifier2 = xgb.XGBClassifier()
    if isGridSearch:
        # Grid search
        # On split 80 / 20 les donnees

        grid_search = gridSearch(xgboost_classifier2, X2_train, y2_train)
        # We retrain the model with the new parameters
        best_xgboost_classifier = grid_search.best_estimator_
        y_probs = best_xgboost_classifier.predict_proba(X2_test)

        y2_test = pd.Series(y2_test)
        CLFS = [[[best_xgboost_classifier], X2_test, 'Best XGBoost GridSearch']]
        Ys = [["Best XGBoost GridSearch", y_probs[:, 1], "blue", True]]

        ROC_plot(y2_test, Ys)
        Centiles_plot(y2_test, Ys)
        cumulative_centiles_plot(y2_test, Ys)
        calibrate_display(CLFS, y2_test)

        experiment.log_parameter("learning_rate", best_xgboost_classifier.learning_rate)
        experiment.log_parameter("max_depth", best_xgboost_classifier.max_depth)
        experiment.log_parameter("n_estimator", best_xgboost_classifier.n_estimators)
        new_best_xgboost_classifier = xgb.XGBClassifier(learning_rate=best_xgboost_classifier.learning_rate, max_depth=best_xgboost_classifier.max_depth, n_estimators=best_xgboost_classifier.n_estimators)
    else:
        new_best_xgboost_classifier = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)

    # Grid search
    x3, y3 = feature_engineering.preprocessing(data, "is_goal")
    X3_train, X3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.2)

    new_best_xgboost_classifier = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=200)
    new_best_xgboost_classifier.fit(X3_train, y3_train)
    
    # 3. Feature
    featureSelectionXgboost = featureSelection(new_best_xgboost_classifier, y3_test, X3_train, y3_train, X3_test)

    # Dump modele
    with open("../data/XGBmodel.pickle", "wb") as outfile:
        pickle.dump(featureSelectionXgboost, outfile)
        outfile.close()

    experiment.log_model('XGBoost', '../data/XGBmodel.pickle')
    experiment.end()

    return featureSelectionXgboost

#
if __name__ == "__main__":
    xgboost(False)