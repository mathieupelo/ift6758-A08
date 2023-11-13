import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import zero
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from feature_engineering import preprocessing

# https://www.kaggle.com/code/henaghonia/fttransformer
# TODO: Add hyperparameters tuning

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


def ann_model(X, y):
    
    # Convertir en tenseur
    y = torch.from_numpy(y).type(torch.float)
    X = torch.from_numpy(X.values).type(torch.float)
    # Combiner X et y dans un Dataset
    train = torch.utils.data.TensorDataset(X, y)

    # Instantier un dataloader
    g = torch.Generator()
    g.manual_seed(8)
    batch_size = 32 # Tradeoff entre computationnal (speed) and good gradient estimate (accuracy)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)

    # Suivant le tutoriel de Pytorch: https://www.learnpytorch.io/02_pytorch_classification/
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Définir le modèle
    nb_features = X.shape[1]
    nb_hidden = 2 # For the moment, is is taking a value between nb_features and nb_output
    nb_output = 1

    class ANNModel(nn.Module):
        def __init__(self):
            super(ANNModel, self).__init__()
            self.layer_1 = nn.Linear(nb_features, nb_hidden)
            self.layer_out = nn.Linear(nb_hidden, nb_output)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(nb_hidden)
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.dropout(x)
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
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, nesterov=True, momentum=0.9) # Fixed momentum
    epochs = 50


    # Entrainer le modèle
    #model.train()
    train_loss = []
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            # Entrainement
            model.train()

            # Forward pass
            y_pred = model(x_batch).squeeze()
            y_pred_probs = torch.sigmoid(y_pred)
        
            # Calculer la perte
            loss = loss_function(y_pred, y_batch)
            accuracy = accuracy_fn(y_batch, torch.round(y_pred_probs))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        train_loss.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {accuracy:.2f}%")

    # Plot function de perte sur l'ensemble d'entrainement
    plt.plot(train_loss)
    plt.xlabel('Epochs')
    plt.xlim(0, epochs)
    plt.xticks(range(0, epochs+1, 1))
    plt.ylabel('Loss')
    plt.savefig('../figures/train_loss.svg')


if __name__ == '__main__':
    df = pd.read_csv('../data/derivatives/train_data.csv')
    X, y = preprocessing(df, 'goalFlag')
    ann_model(X, y)