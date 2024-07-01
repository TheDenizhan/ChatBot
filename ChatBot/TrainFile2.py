import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from preprocessing import bag_of_words, tokenize, stem

from model import TransformerNet, GRUNet, LSTMNet, NeuralNet

# Load intents and prepare data
with open('intentsTR.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)


num_epochs = 2000
batch_size = 32
learning_rate = 0.0001
input_size = len(X[0])
hidden_size = 128
output_size = len(tags)
nhead = 1
k_folds = 5


if input_size % nhead != 0:
    raise ValueError(f"input_size ({input_size}) must be divisible by nhead ({nhead})")

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def train_and_evaluate_model(model, criterion, optimizer, train_loader, test_loader):
    model.train()
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device).unsqueeze(1)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for words, labels in test_loader:
            words = words.to(device).unsqueeze(1)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            outputs = outputs.squeeze(1)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
    return accuracy


dataset = ChatDataset(X, y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = {
    "SimpleNN": NeuralNet(input_size, hidden_size, output_size),
    "LSTM": LSTMNet(input_size, hidden_size, output_size),
    "GRU": GRUNet(input_size, hidden_size, output_size),
    "Transformer": TransformerNet(input_size, hidden_size, output_size, nhead=nhead)
}

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
accuracies = {name: [] for name in models}

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    train_subsampler = Subset(dataset, train_idx)
    test_subsampler = Subset(dataset, test_idx)

    train_loader = DataLoader(dataset=train_subsampler, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_subsampler, batch_size=batch_size, shuffle=False, num_workers=0)

    for name, model in models.items():
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        accuracy = train_and_evaluate_model(model, criterion, optimizer, train_loader, test_loader)
        accuracies[name].append(accuracy)
        print(f"Fold {fold + 1}, {name} model accuracy: {accuracy:.4f}")

print("Model Accuracies:")
for name, acc_list in accuracies.items():
    mean_accuracy = np.mean(acc_list)
    print(f"{name}: {mean_accuracy:.4f}")
