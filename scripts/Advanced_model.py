"""
5. Modèles avancés (20%)
Maintenant que nous avons de nombreuses caractéristiques pour travailler avec, voyons si cela nous permet d’améliorer sur nos modèles de régression logistique simples dans la partie 3.
Nous nous concentrerons sur les modèles XGBoost pour cette section; vous aurez le champ libre pour essayer ce que vous voulez dans la section suivante.

Questions
Pour chacune des questions suivantes, les quatre mêmes figures que dans la partie 3 seront utilisés comme mesures quantitatives :
ROC/AUC
Taux de buts vs percentile de probabilité
Proportion cumulée de buts vs percentile de probabilité
Courbe de fiabilité
"""
import numpy as np
import pandas as pd
from comet_ml import Experiment
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score



"""
1.Entraînez un classificateur XGBoost en utilisant le même ensemble de données en utilisant uniquement les caractéristiques de distance et d' angle (similaire à la partie 3). 
Ne vous inquiétez pas encore du réglage des hyperparamètres, cela servira simplement de comparaison avec la ligne de base avant d'ajouter plus de caractéristiques. 
Ajoutez les courbes correspondantes aux quatre figures à votre article de blog. 
Discutez brièvement (quelques phrases) de votre configuration d'entraînement/validation et comparez les résultats à la référence de régression logistique.
Incluez un lien vers l'entrée comet.ml appropriée pour cette expérience, mais vous n'avez pas besoin de consigner ce modèle dans le registre des modèles.
"""



experiment = Experiment(
    api_key="Bgx9192SVK3nzJNLQcV5nneQS",
    project_name="milestone-2",
    workspace="me-pic"
)



# 1. Importer les nouveaux fichiers
# TODO: Change for the right filepath dataframe_milstone2
data = pd.read_csv('../data/derivatives/dataframe_milestone_2.csv')

print(data.head())
print(data.columns)
# TODO: mettre les caracteristiques de distance et dangle
X = data[['distance_from_last_event', 'changement_angle_tir']]
y = data['goalFlag']


# On split 80 / 20 les donnees
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


xgboost_classifier = xgb.XGBClassifier()
xgboost_classifier.fit(X_train, y_train)

y_pred_prob = xgboost_classifier.predict_proba(X_test)

# ROC
roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])
fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:,1])
average_precision = average_precision_score(y_test, y_pred_prob[:,1])
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob[:,1])

# Enregistrez les mesures dans Comet.ml
experiment.log_metric("ROC AUC", roc_auc)
experiment.log_metric("Average Precision", average_precision)
# Vous pouvez également enregistrer les courbes ROC et PR
experiment.log_curve("ROC Curve", fpr, tpr)
experiment.log_curve("PR Curve", recall, precision)



# centiles
def Centiles_plot(y, y_prob):
    centiles = np.percentile(y_prob, np.arange(0, 101, 5))  # Centiles de 0 à 100 par pas de 10

    # listes pour stocker les taux de buts et les centiles correspondants
    taux_buts = []
    # les probabilités en groupes basés sur les centiles
    for i in range(20):
        lower_bound = centiles[i]
        upper_bound = centiles[i + 1]

        # Filtrer les probabilités dans l'intervalle du centile actuel
        indices = np.where((y_prob >= lower_bound) & (y_prob <= upper_bound))
        # Calculer le taux de buts pour ce groupe

        goal_rate = sum(y.iloc[indices]) / len(y.iloc[indices]) * 100

        # Stocker le taux de buts et le centile correspondant
        taux_buts.append(goal_rate)

    # Tracer le graphique
    plt.plot(np.arange(0, 100, 5), taux_buts, linestyle='-')
    plt.xlabel("Centile de la Probabilité de Tir")
    plt.ylabel("Taux de Buts")
    plt.title("Taux de Buts en fonction du Centile de Probabilité de Tir")
    plt.grid(True)
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))

    plt.show()
    # Log the figure to Comet.ml
    experiment.log_figure(figure=plt, figure_name="Centiles_plot")

Centiles_plot(y_test, y_pred_prob[:,1])


def cumulative_centiles_plot(y, y_prob):
    centiles = np.percentile(y_prob, np.arange(0, 101, 5))  # Centiles de 0 à 100 par pas de 10
    # listes pour stocker les taux de buts et les centiles correspondants
    cumulative_goal_proportion = []
    total_goals = sum(y)
    cumulative_goals = 0
    # les probabilités en groupes basés sur les centiles
    for i in range(20):
        lower_bound = centiles[i]
        upper_bound = centiles[i + 1]

        # Filtrer les probabilités dans la plage du centile actuel
        indices = np.where((y_prob >= lower_bound) & (y_prob <= upper_bound))
        # Calculer le cumule de buts pour ce groupe

        cumulative_goals += sum(y.iloc[indices])

        # Stocker la proportion du cumule de buts
        cumulative_goal_proportion.append((cumulative_goals / total_goals) * 100)
    print("Cumulative_centile done")

    # Plot and log the figure to Comet.ml
    plt.plot(np.arange(0, 100, 5), cumulative_goal_proportion, linestyle='-')
    plt.xlabel("Centile de la Probabilité de Tir")
    plt.ylabel("Proportion")
    plt.title("Cumulatif des buts en %")
    plt.grid(True)
    plt.xticks(np.arange(0, 100, 10))
    plt.yticks(np.arange(0, 100, 10))

    plt.show()
    # Log the figure to Comet.ml
    experiment.log_figure(figure=plt, figure_name="Cumulative Goals Percentage")


cumulative_centiles_plot(y_test, y_pred_prob[:,1])

# TODO: Log hyperparameter sklearn functions
#experiment.log_parameter("")
reshaped_X1_val = X_test.values.reshape(-1,1)
reshaped_y1_val = y_train.values.reshape(-1,1)

# Courbe de calibration
CalibrationDisplay.from_estimator(xgboost_classifier, reshaped_X1_val, reshaped_y1_val, n_bins=30)


"""
2. Maintenant, entraînez un classificateur XGBoost en utilisant toutes les caractéristiques que vous avez créées dans la Partie 4 et effectuez quelques réglages 
d'hyperparamètres pour essayer de trouver le modèle le plus performant avec toutes ces caractéristiques.
 Dans votre article de blog, discutez de votre configuration de réglage des hyperparamètres et incluez des figures pour justifier votre choix d'hyperparamètres. 
 Par exemple, vous pouvez sélectionner les métriques appropriées et effectuer une recherche par grille avec validation croisée. 
 Une fois réglé, intégrez les courbes correspondant au meilleur modèle aux quatre figures de votre article de blog et comparez brièvement les résultats au baseline XGBoost de la premiè`re partie. 
 Incluez un lien vers l'entrée comet.ml appropriée pour cette expérience et enregistrez ce modèle dans le registre des modèles.
"""
# Latin Hypercube search -> Hyperparameter
def find_best_xgb_hyperparams():

    return

"""
Enfin, explorez l'utilisation de certaines techniques de sélection de caractéristiques pour voir si vous pouvez simplifier vos caractéristiques d'entrée. 
Un certain nombre de caractéristiques contiennent des informations corrélées, vous pouvez donc essayer de voir si certaines d'entre elles sont redondantes. 
Vous pouvez essayer certaines des techniques de sélection de caractéristiques discutées en classe; beaucoup d'entre eux sont implémentés pour vous par scikit-learn. 
Vous pouvez également utiliser une librairie comme SHAP pour essayer d'interpréter les caractéristiques sur lesquelles votre modèle repose le plus. 
Discutez des stratégies de sélection de caractéristiques que vous avez essayées et de l'ensemble de caractéristiques le plus optimal que vous avez proposé. 
Incluez quelques figures pour justifier vos affirmations. 
Une fois que vous avez trouvé l'ensemble optimal de caractéristiques via le réglage des hyperparamètres /validation croisée, si l'ensemble de caractéristiques 
est différent de celui utilisé pour la Q2 de cette section, incluez les courbes correspondant au meilleur modèle aux quatre figures de votre article de blog, 
et comparer brièvement les résultats à la référence XGBoost. 
Incluez un lien vers l'entrée comet.ml appropriée pour cette expérience et enregistrez ce modèle dans le registre des modèles.
"""







