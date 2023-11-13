import os
import sys
import timeit
current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))
import pandas as pd
from comet_ml import Experiment
from comet_ml.integration.sklearn import log_model
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score

from feature_engineering import preprocessing
from Basic_model import Centiles_plot, ROC_plot, cumulative_centiles_plot, calibrate_display


def RunHistGB():
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )
    experiment.set_name('HistGradientBoosting')

    print("Loading data...")
    df = pd.read_csv('../data/derivatives/dataframe_milestone_2.csv')
    X, y = preprocessing (df, 'goalFlag')
    CATEGORICAL_FEATURES = ["prd", "noGoalie", "rebond", "shotCategory_Backhand", "shotCategory_Deflected", "shotCategory_Slap Shot", "shotCategory_Snap Shot", "shotCategory_Tip-In", "shotCategory_Wrap-around", "shotCategory_Wrist Shot", "last_event_type_BLOCKED_SHOT", "last_event_type_FACEOFF", "last_event_type_GIVEAWAY", "last_event_type_GOAL", "last_event_type_HIT", "last_event_type_MISSED_SHOT", "last_event_type_PENALTY", "last_event_type_SHOT", "last_event_type_TAKEAWAY"]

    # Recherche des hyperparamètre sur subset de 10% des données (Le calcul est très couteux si on entraine sur toute la base de données 
    # et pour toutes les combinaisons d'hyperparamètres possibles vu la complexité du modèle)

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Ensemble des hyperparamètres
    # Les noyaux gaussienne et plynomiales sont choisit puisque nous savons que nos données sont pas linéairement séparables
    # La valeur de la régularisation 'C' représente la complexité du modèle (de combien nous pénalisons les prédits négatif)
    print("Defining hyperparameters...")
    HP_GB = {'learning_rate': [0.01, 0.1, 0.5], 'max_iter': [100, 200, 300], 'max_depth': [3, 5, 10]}
    hgb_clf = HistGradientBoostingClassifier(categorical_features=CATEGORICAL_FEATURES, scoring='balanced_accuracy', early_stopping=True)

    # Validation croisé aléatoire 
    clf = RandomizedSearchCV(hgb_clf, HP_GB, refit=True, n_iter=10, n_jobs=3)
    clf.fit(X_train, y_train)

    # Prédiction des probabilités de prédire un but
    print("Predicting probabilities...")
    y_pred = clf.predict(X_val)
    y_pred_proba = clf.predict_proba(X_val)

    # Affichage des résultats
    print(classification_report(y_val, y_pred))
    roc_auc = roc_auc_score(y_val, y_pred_proba[:,1]) 
    print(f"ROC AUC Score: {roc_auc}")
    experiment.log_metric('ROC AUC Score', roc_auc)

    Centiles_plot(pd.Series(y_val), pd.Series(y_pred_proba[:,1]), 'HistGradientBoosting')
    ROC_plot(y_val, y_pred_proba[:,1], 'HistGradientBoosting')
    cumulative_centiles_plot(pd.Series(y_val), pd.Series(y_pred_proba[:,1]), 'HistGradientBoosting')
    calibrate_display(clf, X_val, y_val, n_bin=10, model='HistGradientBoosting')

    log_model(
        experiment,
        'HistGradientBoosting',
        clf,
    )
