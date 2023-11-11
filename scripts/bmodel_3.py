import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import comet_ml

data = pd.read_csv('data/derivatives/features_train1.csv')

X_2 = data[['distance_goal','angle_goal']]
y = data['is_goal']
X2_train, X2_val, y2_train, y2_val = train_test_split(
        X_2, y, test_size=0.2, random_state=42
        )
y2_train = y2_train.values.reshape(-1,1)
reshaped_y2_val = y2_val.values.reshape(-1,1)
clf_2 = LogisticRegression().fit(X2_train, y2_train)
y2_score = clf_2.predict_proba(X2_val)



experiment5 = comet_ml.Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='Milestone_2',
    workspace= 'me-pic'
    )


experiment5.log_model("Regression logistique entrainé sur la distance et l'angle", '../scripts/bmodel_3.py')

experiment5.end