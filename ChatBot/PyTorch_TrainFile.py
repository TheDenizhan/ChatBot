import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from preprocessing import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
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


ignore_words = ['?', '.', '!', '\'s', '%', '=', '/', '+', '&', '<', '>', '-', '_', ',', 'a']

all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


X_data = []
y_data = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_data.append(bag)
    label = tags.index(tag)
    y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)


num_epochs = 2000
batch_size = 32
learning_rate = 0.0001
input_size = len(X_data[0])
hidden_size = 128
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):

    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples



kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for fold, (train_index, test_index) in enumerate(kf.split(X_data)):
    print(f'Fold {fold + 1}')


    train_subset = Subset(ChatDataset(X_data, y_data), train_index)
    test_subset = Subset(ChatDataset(X_data, y_data), test_index)

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False, num_workers=0)


    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)


            outputs = model(words)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for words, labels in test_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        fold_accuracies.append(accuracy)
        print(f'Fold {fold + 1}, Accuracy: {accuracy:.2f}%')

    model.train()

print(f'Average accuracy: {np.mean(fold_accuracies):.2f}%')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
