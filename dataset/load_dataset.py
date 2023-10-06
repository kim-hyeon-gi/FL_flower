import datasets
import torch
import torchtext
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchtext.vocab import build_vocab_from_iterator
from torchvision.datasets import CIFAR10

from dataset.reddit_utils import Reddit_Dataset
from utils import Config


def load_dataset(config):
    if config.data == "cifar10":
        if config.iid == "True":
            return load_cifar10_dataset(config)
        else:
            return load_none_iid_cifar10_dataset(config)
    elif config.data == "reddit":
        return load_reddit_dataset(config)


def load_cifar10_dataset(config):
    # Download and transform CIFAR-10 (train and test)
    NUM_CLIENTS = config.num_clients
    BATCH_SIZE = config.batch_size
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    clientset = CIFAR10(
        "./dataset/cifar10", train=True, download=True, transform=transform
    )
    serverset = CIFAR10(
        "./dataset/cifar10", train=False, download=True, transform=transform
    )

    # Split training set into 10 partitions to simulate the individual dataset
    clientset, server_trainset = random_split(
        clientset, [30000, 20000], torch.Generator().manual_seed(42)
    )
    partition_size = len(clientset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(clientset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    client_trainloaders = []
    client_valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        client_trainloaders.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))
        client_valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    server_trainloader = DataLoader(server_trainset, batch_size=BATCH_SIZE)
    server_testloader = DataLoader(serverset, batch_size=BATCH_SIZE)
    return client_trainloaders, client_valloaders, server_trainloader, server_testloader


def load_none_iid_cifar10_dataset(config):
    # Download and transform CIFAR-10 (train and test)
    NUM_CLIENTS = config.num_clients
    BATCH_SIZE = config.batch_size
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    clientset = CIFAR10(
        "./dataset/cifar10", train=True, download=True, transform=transform
    )
    serverset = CIFAR10(
        "./dataset/cifar10", train=False, download=True, transform=transform
    )

    # Split training set into 10 partitions to simulate the individual dataset

    target_idx = [[] for _ in range(10)]
    pretrain_idx = []
    for i in range(10):
        n = 0
        for idx, label in enumerate(clientset.targets):
            if label == i:
                if n < 3000:
                    target_idx[i].append(idx)
                    n = n + 1
                else:
                    pretrain_idx.append(idx)

    server_trainset = Subset(clientset, pretrain_idx)

    # Split each partition into train/val and create DataLoader
    client_trainloaders = []
    client_valloaders = []
    client_label_kind = 6
    data = []
    data_slice = int(300 / client_label_kind)
    for i in range(60):
        for j in range(10):
            data.append(target_idx[j][i * 50 : (i + 1) * 50])

    for i in range(100):
        data_indice = (
            data[6 * i]
            + data[6 * i + 1]
            + data[6 * i + 2]
            + data[6 * i + 3]
            + data[6 * i + 4]
            + data[6 * i + 5]
        )
        dataset = Subset(clientset, data_indice)
        client_trainloaders.append(
            DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        )

    server_trainloader = DataLoader(
        server_trainset, batch_size=BATCH_SIZE, shuffle=True
    )
    server_testloader = DataLoader(serverset, batch_size=BATCH_SIZE)
    return (
        client_trainloaders,
        client_trainloaders,
        server_trainloader,
        server_testloader,
    )


def load_reddit_dataset(config):
    # Download and transform CIFAR-10 (train and test)
    NUM_CLIENTS = config.num_clients
    BATCH_SIZE = config.batch_size

    server_dataset = datasets.load_dataset(
        "text",
        data_files={
            "test": "/home/hyeongikim/Desktop/FL/dataset/reddit/server_data.txt",
        },
    )

    client_dataset = datasets.load_dataset(
        "csv",
        data_files={"train": "/home/hyeongikim/Desktop/FL/dataset/reddit/client.csv"},
    )
    print(len(client_dataset["train"][0]))
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    tokenize_data = lambda example, tokenizer: {"tokens": tokenizer(example["text"])}
    server_tokenized_dataset = server_dataset.map(
        tokenize_data, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer}
    )

    vocab = torchtext.vocab.build_vocab_from_iterator(
        server_tokenized_dataset["test"]["tokens"], min_freq=4
    )
    vocab.insert_token("<unk>", 0)
    vocab.insert_token("<eos>", 1)
    vocab.set_default_index(vocab["<unk>"])
    config.__dict__["vocab_size"] = len(vocab)
    server_train_data = reddit_processing_data(server_tokenized_dataset["test"], vocab)
    length = server_train_data.shape[0]
    server_train_data = server_train_data[: length - (length + 1) % 50]

    client_trainloaders = []
    client_testloaders = []
    for i in range(NUM_CLIENTS):
        client_data = client_dataset["train"][i]["text"].split("\n")
        tokenized_data = []
        for j in client_data:
            tokenized_data.extend(tokenizer(j))
            tokenized_data.append("<eos>")
        tokens = [vocab[token] for token in tokenized_data]
        data = torch.LongTensor(tokens)
        length = data.shape[0]
        data = data[: length - (length + 1) % 50]
        data = Reddit_Dataset(data)
        len_te = len(data) // 10
        len_tr = len(data) - len_te
        client_train, client_test = random_split(
            data, [len_tr, len_te], torch.Generator().manual_seed(42)
        )
        client_trainloaders.append(DataLoader(client_train, batch_size=BATCH_SIZE))
        client_testloaders.append(DataLoader(client_test, batch_size=BATCH_SIZE))
    server_train_dataset = Reddit_Dataset(server_train_data)
    len_te = len(server_train_dataset) // 10
    len_tr = len(server_train_dataset) - len_te
    server_train, server_test = random_split(
        server_train_dataset, [len_tr, len_te], torch.Generator().manual_seed(42)
    )
    server_trainloader = DataLoader(server_train, batch_size=BATCH_SIZE)
    server_testloader = DataLoader(server_test, batch_size=BATCH_SIZE)

    # for i in range(server_train_dataset.__len__()):
    #     print(server_train.__getitem__(i).shape)

    return (
        client_trainloaders,
        client_testloaders,
        server_trainloader,
        server_testloader,
    )


def reddit_processing_data(dataset, vocab):
    data = []
    for example in dataset:
        if example["tokens"]:
            tokens = example["tokens"].append("<eos>")
            tokens = [vocab[token] for token in example["tokens"]]
            data.extend(tokens)
    data = torch.LongTensor(data)
    return data
