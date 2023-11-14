import os
import sys
import pickle
current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))

from Plots import Centiles_plot, ROC_plot, cumulative_centiles_plot, calibrate_display
from comet_ml import Experiment
from comet_ml.integration.sklearn import log_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def runRegression():
    
    data = pd.read_csv('../data/derivatives/features_train1.csv')
    print('hna2')
    X = data[['distance_goal','angle_goal']]
    y = data['is_goal']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic regression trained on 'distance' feature

    X1_train = X_train['distance_goal']
    X1_val = X_val['distance_goal']

    # Reshape to the input shape of LogisticRegrssion() model of sklearn

    X1_train = X1_train.values.reshape(-1,1)
    X1_val = X1_val.values.reshape(-1,1)

    y_train = y_train.values.reshape(-1,1)
    reshaped_y_val = y_val.values.reshape(-1,1)



    clf_1 = LogisticRegression().fit(X1_train, y_train)
    y1_pred = clf_1.predict(X1_val)    

    # Accuracy of model predictions 

    print(f'Accuracy = {accuracy_score(y_val, y1_pred)}')
    print(f'Matrice de confusion = \n {confusion_matrix(y_val, y1_pred)}')

    # Use the probability of prediction

    y1_prob = clf_1.predict_proba(X1_val)
    print (y1_prob)

    # The first column represents the probability that the model does not predict a goal for the corresponding input row. 
    # The second column represents the probability that the model predicts a goal for the corresponding input row.

    # Inputs of plots functions

    Y=[["Logistic regression using Distance", y1_prob[:,1], "blue", True]]
    CLF = [[[clf_1], X1_val, 'Logistic_reg_Distance']]

    AUC = ROC_plot(y_val, Y)
    Centiles_plot(y_val, Y)
    cumulative_centiles_plot(y_val, Y)
    calibrate_display(CLF, y_val, n_bin = 50)

    # Experiment for first model
    
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )

    experiment.set_name('Logistic_reg_dist')

    experiment.log_metric('ROC AUC Score', AUC['Logistic regression using Distance'])

    # Dump modele
    with open("../models/Logistic_reg_distance.pickle", "wb") as outfile:
        pickle.dump(clf_1, outfile)
        outfile.close()
    
    # Log the saved model

    experiment.log_model('Logistic_reg_distance', '../models/Logistic_reg_distance.pickle')

    # Logistic regression trained on 'angle' feature

    X2_train = X_train['angle_goal']
    X2_val = X_val['angle_goal']

    X2_train = X2_train.values.reshape(-1,1)
    X2_val = X2_val.values.reshape(-1,1)

    experiment_1 = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )

    experiment_1.set_name('Logistic_reg_angle')

    

    clf_2 = LogisticRegression().fit(X2_train, y_train)
    y2_prob = clf_2.predict_proba(X2_val)

    with open("../models/Logistic_reg_angle.pickle", "wb") as outfile:
        pickle.dump(clf_2, outfile)
        outfile.close()

    experiment_1.log_model('Logistic_reg_angle', '../models/Logistic_reg_angle.pickle')

    # Logistic regression trained on 'ditance' and 'angle' features

    clf_3 = LogisticRegression().fit(X_train, y_train)
    y3_prob = clf_3.predict_proba(X_val)

    with open("../models/Logistic_reg_dist-angle.pickle", "wb") as outfile:
        pickle.dump(clf_3, outfile)
        outfile.close()

    # Random bas line used to visualize if the other prediction are similarly random or not

    y_uniform_sampled = np.random.uniform(0, 1, len(y_val))

    # Inputs for multiple curve's plots functions

    CLFS = [[[clf_1], X1_val, 'Logistic_reg_Distance'], 
        [[clf_2], X2_val, 'Logistic_reg_Angle'], 
        [[clf_3], X_val, 'Logistic_reg_Distance-Angle'], 
        [[y_val, y_uniform_sampled], X_val, 'Ligne de base aléatoire']
        ]
    
    Ys=[["Logistic regression using Distance", y1_prob[:,1], "blue", False],
        ["Logistic regression using Angle", y2_prob[:,1], "orange", False],
        ["Logistic regression using Distance and Angle", y3_prob[:,1], "green", False],
        ["Ligne de base aléatoire", y_uniform_sampled, "red", True]
        ]
    
    AUCs = ROC_plot(y_val, Ys)
    Centiles_plot(y_val, Ys)
    cumulative_centiles_plot(y_val, Ys)
    calibrate_display(CLFS, y_val, n_bin = 40)

    experiment_1.log_metric('ROC AUC Score', AUCs['Logistic regression using Angle'])

    experiment_2 = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='Milestone_2',
        workspace='me-pic',
    )

    experiment_2.set_name('Logistic_reg_dist_angle')
    experiment_2.log_model('Logistic_reg_dist-angle', '../models/Logistic_reg_dist-angle.pickle')
    experiment_2.log_metric('ROC AUC Score', AUCs['Logistic regression using Distance and Angle'])
    return clf_1, clf_2, clf_3

if __name__ == "__main__":
    runRegression()