import numpy as np
import torch

from dataset.load_dataset import load_dataset
from model.cifar_base import Net
from model.nwp_lstm import LSTM

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def train(net, trainloader, device, epochs: int, verbose=True):
#     DEVICE = device
#     """Train the network on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters())
#     net.to(device)
#     net.train()
#     for epoch in range(epochs):
#         correct, total, epoch_loss = 0, 0, 0.0
#         for images, labels in trainloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = net(images)

#             prediction = outputs.reshape(labels.size(0) * 50, -1)
#             labels = labels.reshape(-1)

#             loss = criterion(prediction, labels)
#             loss.backward()
#             optimizer.step()
#             # Metrics
#             epoch_loss += loss
#             total += labels.size(0)
#             correct += (torch.max(prediction.data, 1)[1] == labels).sum().item()
#         epoch_loss /= len(trainloader.dataset)
#         epoch_acc = correct / total

#         if verbose:
#             print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


# def test(net, testloader, device):
#     """Evaluate the network on the entire test set."""
#     DEVICE = device
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, total, loss = 0, 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = net(images)
#             prediction = outputs.reshape(labels.size(0) * 50, -1)
#             labels = labels.reshape(-1)
#             loss += criterion(prediction, labels).item()
#             _, predicted = torch.max(prediction.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     loss /= len(testloader.dataset)
#     accuracy = correct / total
#     return loss, accuracy


def load_model(config):
    if config.model == "Net":
        return Net()
    elif config.model == "lstm":
        return LSTM(
            config.vocab_size,
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            config.dropout_rate,
            config.tie_weights,
        )


def model_pretraining(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _, _, server_trainloader, _ = load_dataset(config)
    model = load_model(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params:,} trainable parameters")
    print("SEVER Pretraining start")
    train(model, server_trainloader, device, config.server_pretrain_epoch)
    print("SEVER Pretraining end")
    path = "/home/hyeongikim/Desktop/FL/model_pt/" + config.model + ".pt"
    torch.save(model, path)
    config.__dict__["model_pt_path"] = path
    return path


# cifar용


def train(net, trainloader, device, epochs: int, verbose=True):
    DEVICE = device
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.to(device)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, device):
    """Evaluate the network on the entire test set."""
    DEVICE = device
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
