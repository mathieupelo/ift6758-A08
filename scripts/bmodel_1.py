import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import comet_ml


data = pd.read_csv('data/derivatives/features_train1.csv')
X = data['distance_goal']
y = data['is_goal']
X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

# LogisticRegression prend des shape de (n,)
X_train = X_train.values.reshape(-1,1)
y_train = y_train.values.reshape(-1,1)
reshaped_X_val = X_val.values.reshape(-1,1)
reshaped_y_val = y_val.values.reshape(-1,1)

clf = LogisticRegression().fit(X_train, y_train)


experiment1 = comet_ml.Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='Milestone_2',
    workspace= 'me-pic'
    )

experiment1.log_model(name="Regression logistique entrainé sur la distance", file_or_folder='../scripts/bmodel_1')

experiment1.end()