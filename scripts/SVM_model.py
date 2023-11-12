import os
import sys
current_dir = os.path.abspath(os.getcwd())
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'scripts'))
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV

from feature_engineering import preprocessing




df = pd.read_csv('../data/derivatives/dataframe_milestone_2.csv')

X, y = preprocessing (df, 'goalFlag')

# Recherche des hyperparamètre sur subset de 10% des données (Le calcul est très couteux si on entraine sur toute la base de données 
# et pour toutes les combinaisons d'hyperparamètres possibles vu la complexité du modèle)

X1_subset, X1, y1_subset, y1 = train_test_split(X, y, test_size=0.9, random_state=42)


# Ensemble des hyperparamètres
# Les noyaux gaussienne et plynomiales sont choisit puisque nous savons que nos données sont pas linéairement séparables
# La valeur de la régularisation 'C' représente la complexité du modèle (de combien nous pénalisons les prédits négatif)

parameters = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 0.8], 'degree': [2, 3]}
svc = svm.SVC(probability= True)

# Validation croisé aléatoire 

clf1 = RandomizedSearchCV(svc, parameters, refit=True, n_iter=10, n_jobs=-1)
clf1.fit(X1_subset, y1_subset)


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42)

# Entrainement avec les hyperparamètres sélectionnés

parameters = clf1.best_params_
clf_final = svm.SVC( **parameters, probability= True, cache_size=2000)
clf_final.fit(X1_train, y1_train)

# Prédiction des probabilités de prédire un but

y1_score = clf_final.predict_proba(X1_test)
#test