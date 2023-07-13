import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np

# import matplotlib.pyplot as plt
# import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from client.client import FlowerClient
from dataset.load_dataset import load_dataset
from model.utils import load_model
from server.base_strategy import base_strategy
from server.Fed_Custom import FedCustom
from utils import Config, get_parameters, set_parameters

# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# from flwr.common import Metrics
# from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import CIFAR10


# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)


def main(parser):
    # Config
    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    # tensorboard
    writer = SummaryWriter()
    config = Config(json_path=config_dir / "reddit_config.json")
    print(
        f"Training on {config.device} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )
    NUM_CLIENTS = config.num_clients
    BATCH_SIZE = config.batch_size

    client_resources = {"num_cpus": 24}
    if config.device == "cuda":
        client_resources = {"num_gpus": 1}
    trainloaders, valloaders, _, server_testloader = load_dataset(config)

    def client_fn(cid) -> FlowerClient:
        net = load_model(config).to(config.device)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=FedCustom(config, server_testloader, writer=writer),
        client_resources=client_resources,
    )
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        default="config",
        help="Directory containing config.json of data",
    )

    main(parser)
