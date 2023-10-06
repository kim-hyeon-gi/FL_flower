import random

import numpy as np
import torch

from dataset.load_dataset import load_dataset
from model.cifar_base import FunctionNet, Net
from model.nwp_lstm import LSTM

# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(config):
    if config.model_name == "Net":
        return Net()
    elif config.model_name == "lstm":
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
    train(
        model,
        server_trainloader,
        device,
        config.server_pretrain_epoch,
        config.model_name,
        0.001,
    )
    print("SEVER Pretraining end")
    path = "/home/hyeongikim/Desktop/FL/model_pt/" + config.model_name + ".pt"
    torch.save(model, path)
    config.__dict__["model_pt_path"] = path
    return path


def train(model, trainloader, device, epochs, model_name, lr):
    if model_name == "Net":
        cifar10_train(model, trainloader, device, epochs, lr)
    else:
        reddit_train(model, trainloader, device, epochs, lr)


def low_rank_train(
    model, trainloader, device, epochs, model_name, lr, structure, shape_2d, p_value
):
    if model_name == "Net":
        return low_rank_cifar10_train(
            model, trainloader, device, epochs, lr, structure, shape_2d, p_value
        )
    else:
        pass
        # low_rank_reddit_train(model, trainloader, device, epochs, lr)


def test(model, testloader, device, model_name):
    if model_name == "Net":
        return cifar10_test(model, testloader, device)
    else:
        return reddit_test(model, testloader, device)


def reddit_train(net, trainloader, device, epochs: int, lr):
    DEVICE = device
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    net.to(device)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)

            prediction = outputs.reshape(labels.size(0) * 50, -1)
            labels = labels.reshape(-1)

            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(prediction.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def reddit_test(net, testloader, device):
    """Evaluate the network on the entire test set."""
    DEVICE = device
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            prediction = outputs.reshape(labels.size(0) * 50, -1)
            labels = labels.reshape(-1)
            loss += criterion(prediction, labels).item()
            _, predicted = torch.max(prediction.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


# cifar용


def cifar10_train(net, trainloader, device, epochs: int, lr):
    DEVICE = device
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def cifar10_test(net, testloader, device):
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


def low_rank_cifar10_train(
    net, trainloader, device, epochs: int, lr, structure, shape_2d, p_value
):
    DEVICE = device
    rand = [random.randint(1, 9999) for _ in range(len(shape_2d))]
    U = [
        torch.randn(
            (size[0], p),
            generator=torch.Generator().manual_seed(n),
            requires_grad=False,
        ).to("cuda")
        for size, p, n in zip(shape_2d, p_value, rand)
    ]
    V = []
    i = 0
    for name, child in net.features.named_children():
        for param in child.parameters():
            tensor = param.data
            tensor = tensor.view(shape_2d[i])
            pinverse = torch.mm(torch.linalg.inv(torch.mm(U[i].T, U[i])), U[i].T)
            V.append(torch.tensor(torch.mm(pinverse, tensor).data, requires_grad=True))
            i = i + 1

    filter = [torch.mm(u, v).view(size) for u, v, size in zip(U, V, structure)]
    i = 0
    for name, child in net.features.named_children():
        for param in child.parameters():
            if i in [0, 1, 16, 17]:
                V[i] = torch.tensor(param.data, requires_grad=True)
            i = i + 1

    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(V, lr=lr)
    net.to(device)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = FunctionNet(images, filter, V)

            loss = criterion(outputs, labels)
            loss.backward(retain_graph=True)

            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return V, rand


def low_rank_cifar10_train2(
    net, trainloader, device, epochs: int, lr, structure, shape_2d, p_value
):
    DEVICE = device
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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

        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    rand = [random.randint(1, 9999) for _ in range(len(shape_2d))]
    U = [
        torch.randn(
            (size[0], p),
            generator=torch.Generator().manual_seed(n),
            requires_grad=False,
        ).to("cuda")
        for size, p, n in zip(shape_2d, p_value, rand)
    ]
    W = [val.to("cuda") for _, val in net.state_dict().items()]
    V = []
    for u, w, size in zip(U, W, shape_2d):
        pinverse = torch.mm(torch.linalg.inv(torch.mm(u.T, u)), u.T)
        w = w.view(size)
        V.append(torch.mm(pinverse, w))
    for i in [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17]:
        V[i] = W[i]
    return V, rand
