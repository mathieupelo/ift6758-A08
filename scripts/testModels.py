from sklearn.metrics import roc_auc_score

from scripts.Plots import *

def testModels(models, test_df):
    test_df2 = test_df
    X_test = test_df2.drop('goalFlag', axis=1)
    y_test = test_df['goalFlag']

    CLFS = []
    Ys = []

    colors = ["blue", "green", "orange", "red", "black"]

    for idx, model in enumerate(models, start=1):
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        CLFS.append([model], X_test, str(idx))
        Ys.append(str(idx), y_pred_proba, colors[idx], False)

    AUCs = ROC_plot(y_test, Ys)
    Centiles_plot(y_test, Ys)
    cumulative_centiles_plot(y_test, Ys)
    calibrate_display(CLFS, y_test, n_bin=40)

