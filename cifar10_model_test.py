import argparse

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from dataset.load_dataset import load_dataset
from model.cifar_base import Net
from utils import Config


def main():

    # Config
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(
        "./dataset/cifar10", train=True, download=True, transform=transform
    )
    testset = CIFAR10(
        "./dataset/cifar10", train=False, download=True, transform=transform
    )
    clientset, server_trainset = random_split(
        trainset, [30000, 20000], torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(server_trainset, batch_size=256)
    test_loader = DataLoader(testset, batch_size=256)

    net = torch.load("/home/hyeongikim/Desktop/FL/model_pt/Net_acc70.pt")
    device = "cuda"
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.to(device)

    net.train()
    epochs = 1000
    best_acc = 0

    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    if best_acc < accuracy:
        best_acc = accuracy
        path = "/home/hyeongikim/Desktop/FL/model_pt/Net.pt"
        torch.save(net, path)
    print(
        f"Epoch {1}: test loss {loss}, accuracy {accuracy}        best acc : {best_acc}"
    )

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        net.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

        if epoch % 5 == 0:
            correct, total, loss = 0, 0, 0.0
            net.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            loss /= len(test_loader.dataset)
            accuracy = correct / total
            if best_acc < accuracy:
                best_acc = accuracy
                path = "/home/hyeongikim/Desktop/FL/model_pt/Net.pt"
                torch.save(net, path)
            print(
                f"Epoch {epoch+1}: test loss {loss}, accuracy {accuracy}        best acc : {best_acc}"
            )


if __name__ == "__main__":

    main()
