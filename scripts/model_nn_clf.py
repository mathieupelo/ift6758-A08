import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# TODO: Add hyperparameters tuning

# Importer les données
df = pd.read_csv('../data/derivatives/features_train1.csv')
df_copy = df.copy()

# Définir la cible
target = 'is_goal'

# Séparer en X et y
y = df_copy[target]
X = df_copy.drop(columns=target)

# Standardiser les caractéristiques continues
continuous_feat = ['distance_goal', 'angle_goal']
feat = X[continuous_feat]
scaler = StandardScaler().fit(feat.values)
feat = scaler.transform(feat.values)
X[continuous_feat] = feat

# Convertir en tenseur
y = torch.from_numpy(y.values).type(torch.float)
X = torch.from_numpy(X.values).type(torch.float)
# Combiner X et y dans un Dataset
train = torch.utils.data.TensorDataset(X, y)

# Instantier un dataloader
batch_size = 32
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

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
epochs = 20

# Entrainer le modèle
model.train()
train_loss = []
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(x_batch)
        loss = loss_function(y_pred, y_batch.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss.append(loss.item())

plt.plot(train_loss)
plt.xlabel('Epochs')
plt.xlim(0, epochs)
plt.xticks(range(0, epochs+1, 1))
plt.ylabel('Loss')
plt.savefig('../figures/train_loss.svg')