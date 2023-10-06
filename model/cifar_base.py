import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6, 6),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def FunctionNet(x, filter, V):
    x = F.conv2d(x, V[0], bias=V[1], stride=1, padding=1)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, filter[2], bias=V[3], stride=1, padding=1)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, filter[4], bias=V[5], stride=2, padding=1)
    x = F.relu(x, inplace=True)
    x = F.dropout(x, 0.5)
    x = F.conv2d(x, filter[6], bias=V[7], stride=1, padding=1)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, filter[8], bias=V[9], stride=1, padding=1)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, filter[10], bias=V[11], stride=2, padding=1)
    x = F.relu(x, inplace=True)
    x = F.dropout(x, 0.5)
    x = F.conv2d(x, filter[12], bias=V[13], stride=1, padding=0)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, filter[14], bias=V[15], stride=1, padding=0)
    x = F.relu(x, inplace=True)
    x = F.conv2d(x, V[16], bias=V[17], stride=1, padding=0)
    x = F.relu(x, inplace=True)
    x = F.avg_pool2d(x, (6, 6))
    x = x.view(x.size(0), -1)
    return x
