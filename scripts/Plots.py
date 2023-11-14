import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay , roc_curve, auc
from sklearn.calibration import CalibrationDisplay




def ROC_plot (y, Y):
    curves = []
    AUC = {}
    fig, ax = plt.subplots()
    for model, pred, color, chance_lev in Y :
        curves.append(model) 
        RocCurveDisplay.from_predictions(
        y,
        pred,
        name= model,
        color= color,
        plot_chance_level= chance_lev,
        ax= ax,
        )
        fpr, tpr, _ = roc_curve(np.array(y), pred, pos_label=1)
        AUC[model]= auc(fpr, tpr) 
    print ( f'AUC metric : {AUC}')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("But-vs-NBut ROC curves")
    ax.legend()
    plt.savefig(f"../figures/ROC_curve_{[model for model in curves]}.png")
    plt.show()
    
    return AUC
    


def Centiles_plot(y, Y):

    fig, ax = plt.subplots()
    curves = []
    for model, pred, _, __ in Y:
        centiles = np.percentile(pred, np.arange(0, 101, 5))  # Centiles de 0 à 100 par pas de 10
        taux_buts = []
        curves.append(model)
        for i in range(20):
             
            lower_bound = centiles[i]
            upper_bound = centiles[i + 1]
            
            indices = np.where((pred >= lower_bound) & (pred <= upper_bound))
            goal_rate = (sum(y.iloc[indices]) / len(y.iloc[indices])) * 100
            taux_buts.append(goal_rate)

        ax.plot(np.arange(0, 100, 5), taux_buts, linestyle='-', label = model)
    ax.set_xlabel("Centile de la Probabilité de Tir")
    ax.set_ylabel("Taux de Buts")
    ax.set_title("Taux de Buts en fonction du Centile de Probabilité de Tir")
    ax.grid(True)
    ax.set_xlim(100, 0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()
    plt.savefig(f"../figures/Centiles_plot_{[model for model in curves]}.png")
    plt.show()

def cumulative_centiles_plot(y, Y):
    n = len(y)
    curves = []
    fig, ax = plt.subplots()
    for model, pred, _, __ in Y:
        curves.append(model)
        x_axis = np.arange(n)[::-1] * (100 / n)
        reverse_prob = pred[::-1]
        reverse_prob[::-1].sort()
        cum_percentile = np.cumsum(reverse_prob) * 100
        ax.plot(x_axis, cum_percentile / sum(pred), label = model)
    ax.set_xlabel("Centile de la Probabilité de Tir")
    ax.set_ylabel("Proportion")
    ax.set_title(" cumulatif des buts en %")
    ax.grid(True)
    ax.set_xlim(100, 0)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()
    plt.savefig(f"../figures/cumulative_centiles_plot_{[model for model in curves]}.png")
    plt.show()


def calibrate_display(classifier, y_val, n_bin = 40):
    fig, ax = plt.subplots()
    curves = []
    for model, X, name in classifier: 
        curves.append(name)
        if len(model) != 2:

            CalibrationDisplay.from_estimator(model[0], X, y_val, n_bins=n_bin, name=name, ax= ax )
            
        else :
            CalibrationDisplay.from_predictions(model[0], model[1], n_bins= n_bin, name= name, ax= ax)
        
    ax.legend()
    plt.savefig(f"../figures/calibrate_display_{curves}.png")
    plt.show()