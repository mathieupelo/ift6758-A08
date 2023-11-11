import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import comet_ml

data = pd.read_csv('data/derivatives/features_train1.csv')

X_1 = data['angle_goal']
y = data['is_goal']
X1_train, X1_val, y1_train, y1_val = train_test_split(
        X_1, y, test_size=0.2, random_state=42
        )

X1_train = X1_train.values.reshape(-1,1)
y1_train = y1_train.values.reshape(-1,1)
reshaped_X1_val = X1_val.values.reshape(-1,1)
reshaped_y1_val = y1_val.values.reshape(-1,1)

clf_1 = LogisticRegression().fit(X1_train, y1_train)

y1_score = clf_1.predict_proba(reshaped_X1_val)

experiment2 = comet_ml.Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='Milestone_2',
    workspace= 'me-pic'
    )

experiment2.log_model(name="Regression logistique entrainé sur l'angle", file_or_folder='../scripts/bmodel_2.py')

experiment2.end()