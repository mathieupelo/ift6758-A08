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
from sklearn.model_selection import train_test_split

"""
1.Entraînez un classificateur XGBoost en utilisant le même ensemble de données en utilisant uniquement les caractéristiques de distance et d' angle (similaire à la partie 3). 
Ne vous inquiétez pas encore du réglage des hyperparamètres, cela servira simplement de comparaison avec la ligne de base avant d'ajouter plus de caractéristiques. 
Ajoutez les courbes correspondantes aux quatre figures à votre article de blog. 
Discutez brièvement (quelques phrases) de votre configuration d'entraînement/validation et comparez les résultats à la référence de régression logistique.
Incluez un lien vers l'entrée comet.ml appropriée pour cette expérience, mais vous n'avez pas besoin de consigner ce modèle dans le registre des modèles.
"""

import pandas as pd

# 1. Importer les nouveaux fichiers
# TODO: Change for the right filepath
data = pd.read_csv('../data/derivatives/features_train1.csv')

# TODO: mettre les caracteristiques de distance et dangle
X = data[['distance', 'angle']]
y = data['is_goal']


# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
2. Maintenant, entraînez un classificateur XGBoost en utilisant toutes les caractéristiques que vous avez créées dans la Partie 4 et effectuez quelques réglages 
d'hyperparamètres pour essayer de trouver le modèle le plus performant avec toutes ces caractéristiques.
 Dans votre article de blog, discutez de votre configuration de réglage des hyperparamètres et incluez des figures pour justifier votre choix d'hyperparamètres. 
 Par exemple, vous pouvez sélectionner les métriques appropriées et effectuer une recherche par grille avec validation croisée. 
 Une fois réglé, intégrez les courbes correspondant au meilleur modèle aux quatre figures de votre article de blog et comparez brièvement les résultats au baseline XGBoost de la premiè`re partie. 
 Incluez un lien vers l'entrée comet.ml appropriée pour cette expérience et enregistrez ce modèle dans le registre des modèles.
"""


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






