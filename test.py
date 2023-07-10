import math
from glob import iglob

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
tokenize_data = lambda example, tokenizer: {"tokens": tokenizer(example["text"])}
tokenized_dataset = dataset.map(
    tokenize_data, remove_columns=["text"], fn_kwargs={"tokenizer": tokenizer}
)

vocab = torchtext.vocab.build_vocab_from_iterator(
    tokenized_dataset["train"]["tokens"], min_freq=3
)
vocab.insert_token("<unk>", 0)
vocab.insert_token("<eos>", 1)
vocab.set_default_index(vocab["<unk>"])


def get_data(dataset, vocab, batch_size):
    data = []
    for example in dataset:
        if example["tokens"]:
            tokens = example["tokens"].append("<eos>")
            tokens = [vocab[token] for token in example["tokens"]]
            data.extend(tokens)
    data = torch.LongTensor(data)
    num_batches = data.shape[0] // batch_size
    data = data[: num_batches * batch_size]
    data = data.view(batch_size, num_batches)
    return data


batch_size = 128

train_data = get_data(tokenized_dataset["train"], vocab, batch_size)
# valid_data = get_data(tokenized_dataset["validation"], vocab, batch_size)
# test_data = get_data(tokenized_dataset["test"], vocab, batch_size)
# print(np.shape(test_data))


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        tie_weights,
    ):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            assert embedding_dim == hidden_dim, "cannot tie, check dims"
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hidden_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(
                self.embedding_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)
            self.lstm.all_weights[i][1] = torch.FloatTensor(
                self.hidden_dim, self.hidden_dim
            ).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell


vocab_size = len(vocab)
embedding_dim = 64  # 400 in the paper
hidden_dim = 64  # 1150 in the paper
num_layers = 2  # 3 in the paper
dropout_rate = 0.2
tie_weights = True
lr = 1e-3

model = LSTM(
    vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights
).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {num_params:,} trainable parameters")


def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx : idx + seq_len]
    target = data[:, idx + 1 : idx + seq_len + 1]
    return src, target


def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):

    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, : num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    for idx in tqdm(
        range(0, num_batches - 1, seq_len), desc="Training: ", leave=False
    ):  # The last batch can't be a src
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, idx)
        print(np.shape(src))
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)

        prediction = prediction.reshape(batch_size * seq_len, -1)
        target = target.reshape(-1)
        loss = criterion(prediction, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def evaluate(model, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, : num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]
    correct = 0
    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size = src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            prediction = torch.argmax(prediction, dim=-1)
            correct += (prediction == target).sum().item()
            epoch_loss += loss.item() * seq_len
    correct = correct / (num_batches)
    print(correct)
    return epoch_loss / num_batches


n_epochs = 50
seq_len = 50
clip = 0.25
saved = False

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)


best_valid_loss = float("inf")

for epoch in range(n_epochs):
    train_loss = train(
        model, train_data, optimizer, criterion, batch_size, seq_len, clip, device
    )


def generate(
    prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None
):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab["<unk>"]:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab["<eos>"]:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens
