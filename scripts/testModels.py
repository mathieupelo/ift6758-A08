from comet_ml import API
import pandas as pd
import os
import sys
import pickle
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))

from Plots import Centiles_plot, ROC_plot, cumulative_centiles_plot, calibrate_display
from feature_engineering import preprocessing


def visualisation(model_name, best_model, y_probs, y_test, X_test):
    y_test = pd.Series(y_test)
    CLFS = [[[best_model], X_test, model_name]]
    Ys = [[model_name, y_probs[:, 1], "blue", True]]

    print(f"Résultats pour le modèle : {model_name}")

    ROC_plot(y_test, Ys)
    Centiles_plot(y_test, Ys)
    cumulative_centiles_plot(y_test, Ys)
    calibrate_display(CLFS, y_test, n_bin=50)

def runTestModels():
    test_data1 = pd.read_csv('../data/derivatives/test_data_saison_eli.csv')
    test_data2 = pd.read_csv('../data/derivatives/test_data_saison_reg.csv')

    os.environ['api_key'] = 'Bgx9192SVK3nzJNLQcV5nneQS'
    os.environ['COMET_API_KEY'] = 'Bgx9192SVK3nzJNLQcV5nneQS'
    API_KEY = os.environ.get('COMET_API_KEY')
    project_name = "milestone-2"
    workspace = "me-pic"

    api = API(api_key=API_KEY)

    model_list = [
        ("xgboost", "1.4.0", 'full'),  # 'full' signifie utiliser test_data1 et test_data2
        ("logistic_reg_angle", "1.1.0", 'angle'),  # 'logistic' signifie utiliser test_logistic
        ('logistic_reg_dist-angle', "1.1.0", 'dist-angle'),
        ('logistic_reg_distance', "1.1.0", 'distance'),
        ('boosted_logistic_reg', '1.0.0', 'full')
    ]

    test_datasets = [('test_data1', test_data1), ('test_data2', test_data2)]

    for model_name, model_version, data_type in model_list:
        try:
            output_path = f"../model/{model_name}"
            api.download_registry_model(workspace, model_name, model_version, output_path=output_path, expand=True)
            file_name = os.listdir(f'../model/{model_name}')

            with open(os.path.join(output_path, file_name[0]), 'rb') as file:
                model = pickle.load(file)

            for dataset_name, dataset in test_datasets:
                if data_type == 'full':
                    X, y = preprocessing(dataset, 'is_goal')
                elif data_type == 'angle':
                    X, y = preprocessing(dataset, 'is_goal')
                    X = X[['shot_angle']]
                elif data_type == 'distance':
                    X, y = preprocessing(dataset, 'is_goal')
                    X = X[['shot_distance']]
                elif data_type == 'dist-angle':
                    X, y = preprocessing(dataset, 'is_goal')
                    X = X[['shot_distance', 'shot_angle']]

                predictions = model.predict_proba(X)

                # Calcul et affichage des métriques pour chaque modèle et dataset
                metrics = roc_auc_score(y, predictions[:, 1])
                print(f"roc_auc_score for {model_name} on {dataset_name}: {metrics}")

                metrics = average_precision_score(y, predictions[:, 1])
                print(f"average_precision for {model_name} on {dataset_name}: {metrics}")

                metrics = log_loss(y, predictions)
                print(f"log_loss for {model_name} on {dataset_name}: {metrics}")

                metrics = brier_score_loss(y, predictions[:, 1])
                print(f"brier_score for {model_name} on {dataset_name}: {metrics}")

                visualisation(model_name, model, predictions, y, X)

        except Exception as e:
            print(f"An error occurred with {model_name}: {e}")
