import comet_ml
import os
import sys
from joblib import dump, load
current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))
from comet_ml import Experiment
from comet_ml.integration.sklearn import log_model
import pickle
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from feature_engineering import preprocessing
from Plots import Centiles_plot, ROC_plot, cumulative_centiles_plot, calibrate_display


def runBoosted_Logistic_reg():

    df = pd.read_csv('../data/derivatives/train_data.csv')
    X, y = preprocessing (df, 'is_goal')
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Elasticnet regularization combine both L1 penality for irrelevant or redundant features and L2 penality to control stability
    # Only 'saga' solver support this regularization

    clf = LogisticRegression(penalty= 'elasticnet', max_iter= 1000, solver= 'saga', l1_ratio= 0.5).fit(X_train, y_train)

    # Boosting the classifier with XGBoost/ We use the logistic regression mdel as base estimator instead of Dicision tree

    xgb_model = XGBClassifier(base_estimator = clf, random_state=42)

    # Set of hyperparmeter search
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
        'max_depth': [3, 4, 5, 6],
    }

    # Random Cross Validation

    random_sch = RandomizedSearchCV(xgb_model, param_grid, refit=True, n_iter=10, n_jobs=-1)

    # Train the XGBoost model
    random_sch.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = random_sch.best_params_
    print("Best Hyperparameters:", best_params)

    # Make predictions with the best model
    best_model = random_sch.best_estimator_

    y_prob = best_model.predict_proba(X_test)

    Y = [["Boosted_Logistic_reg", y_prob[:,1], "blue", True]]
    CLF = [[[best_model], X_test, 'Boosted_Logistic_reg']]

    AUC = ROC_plot(y_test, Y)
    Centiles_plot(y_test, Y)
    cumulative_centiles_plot(y_test, Y)
    calibrate_display(CLF, y_test, n_bin = 50)

    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )

    experiment.set_name('Boosted_Logistic_reg')

    experiment.log_metric('ROC AUC Score', AUC['Boosted_Logistic_reg'])

    # Dump modele
    with open("../models/Boosted_Logistic_reg.pickle", "wb") as outfile:
        pickle.dump(best_model, outfile)
        outfile.close()
    
    # Log the saved model

    experiment.log_model('Logistic_reg_distance', '../models/Boosted_Logistic_reg.pickle')

    return best_model

if __name__ == "__main__":
    runBoosted_Logistic_reg()
