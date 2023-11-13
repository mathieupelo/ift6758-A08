from sklearn.model_selection import GridSearchCV
from comet_ml import Experiment
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scripts.Basic_model import *

def xgboost():
    def gridSearch(model):
        # Define the hyperparameters to tune and their possible values
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
        grid_search.fit(X_train, y_train)

        # We output the best parameters that were given
        print("Output the best hyperparams")
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters: {best_params}")
        return grid_search

    def featureSelection(best_xgboost_classifier):
        # XGBoost has a L1 regularization term that we can use to identify features that are assigned the 0 weight
        xgboost_classifier = xgb.XGBClassifier(learning_rate=best_xgboost_classifier.learning_rate, n_estimators=best_xgboost_classifier.n_estimators, max_depth=best_xgboost_classifier.max_depth, reg_alpha=1)  # Add reg_alpha for L1 regularization
        xgboost_classifier.fit(X_train, y_train)
        return xgboost_classifier

    data = pd.read_csv('../data/derivatives/train_data.csv')
    experiment = Experiment(
        api_key="Bgx9192SVK3nzJNLQcV5nneQS",
        project_name="milestone-2",
        workspace="me-pic"
    )

    X = data[['distance_from_last_event', 'changement_angle_tir']]
    y = data['goalFlag']

    # On split 80 / 20 les donnees
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xgboost_classifier = xgb.XGBClassifier()
    xgboost_classifier.fit(X_train, y_train)

    y_pred_proba = xgboost_classifier.predict_proba(X_test)[:, 1]

    # Print
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc}")

    ROC_plot(y_test, y_pred_proba)
    # Taux de buts
    Centiles_plot(y_test, y_pred_proba)
    # Cumule de buts
    cumulative_centiles_plot(y_test, y_pred_proba)
    # Calibration display
    calibrate_display(xgboost_classifier, X_test, y_test, 30)



    # Grid search
    grid_search = gridSearch(xgboost_classifier)

    # We retrain the model with the new parameters
    best_xgboost_classifier = grid_search.best_estimator_
    y_pred_proba = best_xgboost_classifier.predict_proba(X_test)[:, 1]

    # We evaluate once again
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc}")

    ROC_plot(y_test, y_pred_proba)

    # Taux de buts
    Centiles_plot(y_test, y_pred_proba)

    # Cumule de buts
    cumulative_centiles_plot(y_test, y_pred_proba)

    # Calibration display
    calibrate_display(xgboost_classifier, X_test, y_test, 30)

    experiment.log_parameter("learning_rate", best_xgboost_classifier.learning_rate)
    experiment.log_parameter("max_depth", best_xgboost_classifier.max_depth)
    experiment.log_parameter("n_estimator", best_xgboost_classifier.n_estimators)

    experiment.log_metric("SOC", roc_auc)

    experiment.log_model("best_xgboost_classifier", best_xgboost_classifier)

    # Feature 
    featureSelectionXgboost = featureSelection(best_xgboost_classifier)
    y_pred_proba = featureSelectionXgboost.predict_proba(X_test)[:, 1]

    # We evaluate once again
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score: {roc_auc}")

    ROC_plot(y_test, y_pred_proba)

    # Taux de buts
    Centiles_plot(y_test, y_pred_proba)

    # Cumule de buts
    cumulative_centiles_plot(y_test, y_pred_proba)

    # Calibration display
    calibrate_display(featureSelectionXgboost, X_test, y_test, 30)













