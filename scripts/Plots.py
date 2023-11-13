import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay , roc_curve, auc
from sklearn.calibration import CalibrationDisplay




def ROC_plot (y, y_prob):
    RocCurveDisplay.from_predictions(
    y,
    y_prob,
    name="Chance d'avoir un but",
    color="darkorange",
    plot_chance_level=True,
    )
    fpr, tpr, _ = roc_curve(np.array(y), y_prob, pos_label=1)
    AUC = auc(fpr, tpr)
    print ( f'AUC metric : {AUC}')
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("But-vs-NBut ROC curves")
    plt.legend()
    plt.show()
    


def Centiles_plot (y, y_prob):
    
    centiles = np.percentile(y_prob, np.arange(0, 101, 10))  # Centiles de 0 à 100 par pas de 10

    # listes pour stocker les taux de buts et les centiles correspondants
    taux_buts = [0]
    # les probabilités en groupes basés sur les centiles
    for i in range(10):
        lower_bound = centiles[i]
        upper_bound = centiles[i + 1]
        
        # Filtrer les probabilités dans l'intervalle du centile actuel
        indices = np.where((y_prob >= lower_bound) & (y_prob <= upper_bound))
        # Calculer le taux de buts pour ce groupe
        
        goal_rate = (sum(y.iloc[indices]) / len(y.iloc[indices]))*100
        
        # Stocker le taux de buts et le centile correspondant
        taux_buts.append(goal_rate)

    # Tracer le graphique
    plt.plot(np.arange(0, 101, 10), taux_buts, linestyle='-')
    plt.xlabel("Centile de la Probabilité de Tir")
    plt.ylabel("Taux de Buts")
    plt.title("Taux de Buts en fonction du Centile de Probabilité de Tir")
    plt.grid(True)
    plt.xlim(100,0)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 101, 10))


    plt.show()

def cumulative_centiles_plot(y, y_prob):

    n=len(y_prob)
    x_axis=np.arange(n)[::-1]*(100/n)
    reverse_prob=y_prob[::-1]
    reverse_prob[::-1].sort()
    cum_percentile=np.cumsum(reverse_prob)*100

    plt.plot(x_axis ,cum_percentile/sum(y_prob))
    plt.xlabel("Centile de la Probabilité de Tir")
    plt.ylabel("Proportion")
    plt.title(" cumulatif des buts en %")
    plt.grid(True)
    plt.xlim(100,0)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 101, 10))


def calibrate_display(classifier, x_val, y_val, n_bin):
    CalibrationDisplay.from_estimator(classifier, x_val, y_val, n_bins=n_bin)