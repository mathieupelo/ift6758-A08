from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import preprocessing
from sklearn.metrics import roc_auc_score

from Basic_model import Centiles_plot, ROC_plot, cumulative_centiles_plot, calibrate_display

# Calculate accuracy (classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

def seed_worker(worker_id):
    """
    References
    ----------
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

experiment = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='Milestone_2',
    workspace='me-pic',
)
experiment.set_name('ANN')

df = pd.read_csv('../data/derivatives/train_data.csv')
X, y = preprocessing(df, 'goalFlag')
    
# Convertir en tenseur
y = torch.from_numpy(y).type(torch.float)
X = torch.from_numpy(X.values).type(torch.float)
# Combiner X et y dans un Dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8, shuffle=True)

# Instantier un dataloader
g = torch.Generator()
g.manual_seed(8)
#batch_size = 32 # Tradeoff entre computationnal (speed) and good gradient estimate (accuracy)
#train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)

# Suivant le tutoriel de Pytorch: https://www.learnpytorch.io/02_pytorch_classification/
# Make device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Définir le modèle
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.layer_1 = nn.Linear(30, 64)
        self.layer_2 = nn.Linear(64, 128)
        self.layer_3 = nn.Linear(128, 96)
        self.layer_4 = nn.Linear(96, 32)
        self.layer_out = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.relu(self.layer_4(x))
        x = self.layer_out(x)
        return x

model = ANNModel().to(device)
print(model)

    # Définir la fonction de perte
    # Puisque nous avons un problème débalancé, nous allons augmenter le poids de la classe
    # positive, ici étant la classe minoritaire correspondant aux buts marqués.
ratio_shots_goals = len(y[y==0]) / len(y[y==1])
loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio_shots_goals))

# Définir le taux d'apprentissage
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) # Fixed momentum
epochs = 500

params = {"num_epochs": epochs, "learning_rate": learning_rate, "momentum": 0.9}
experiment.log_parameters(params)

X_train, X_val = X_train.to(device), X_val.to(device)
y_train, y_val = y_train.to(device), y_val.to(device)

# Entrainer le modèle
epoch_count, train_loss_values, valid_loss_values = [], [], []

for epoch in range(epochs):
    # Entrainement
    model.train()

    # Forward pass
    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    
    # Calculer la perte
    loss = loss_function(y_logits, y_train)
    accuracy = balanced_accuracy_score(y_pred.detach().numpy(), y_train.detach().numpy())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    model.eval()

    with torch.inference_mode():
        valid_logits = model(X_val).squeeze()
        valid_pred = torch.round(torch.sigmoid(valid_logits))

        valid_loss = loss_function(valid_logits, y_val)
        valid_accuracy = balanced_accuracy_score(valid_pred.detach().numpy(), y_val.detach().numpy())

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Training Loss: {loss:.5f}, Training Balanced Accuracy: {accuracy:.2f} | Validation Loss: {valid_loss:.5f}, Validation Balanced Accuracy: {valid_accuracy:.2f}")
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        valid_loss_values.append(valid_loss.detach().numpy())

    # Log metric for each epoch
    experiment.log_metrics({'Training Loss': loss, 'Training Balanced Accuracy': accuracy, 'Validation Loss': valid_loss, 'Validation Balanced Accuracy': valid_accuracy}, epoch=epoch)

# Log model
print('Loggin the model...')
log_model(experiment, model, model_name='ANN')

print('computing metrics on final trained model...')
with torch.inference_mode():
    valid_logits = model(X_val).squeeze()
    valid_pred = torch.round(torch.sigmoid(valid_logits))

roc_auc = roc_auc_score(y_val, valid_pred) 
print(f"ROC AUC Score: {roc_auc}")
experiment.log_metric('ROC AUC Score', roc_auc)

Centiles_plot(pd.Series(y_val), pd.Series(valid_pred), 'ANN')
ROC_plot(y_val, valid_pred, 'ANN')
cumulative_centiles_plot(pd.Series(y_val), pd.Series(valid_pred), 'ANN')

experiment.end()

