# %%
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import IMDB
from transformers import (
    BertModel,
    BertTokenizer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# ## Prepare the IMDB dataset and the BERT model


# %%
# Preprocessing and Tokenization function
def preprocess(batch):
    labels, texts = zip(*batch)
    inputs = tokenizer(
        list(texts), padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    labels = torch.tensor(labels) - 1  # Label 0-indexed for PyTorch
    return inputs, labels


# Load AG News Dataset
train_datapip = IMDB(split="train")  # type: ignore
test_datapip = IMDB(split="test")  # type: ignore

# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)  # type: ignore

batch_size = 128
train_loader = DataLoader(train_datapip, batch_size, True, collate_fn=preprocess)
test_loader = DataLoader(test_datapip, batch_size, collate_fn=preprocess)

# %% [markdown]
# ## Convert text input into BERT embeddings

# %%
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []
with torch.no_grad():
    print("Computing train embeddings")
    for inputs, labels in train_loader:
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        train_embeddings.append(outputs.last_hidden_state.cpu())
        train_labels.append(labels)
    print("Computing test embeddings")
    for inputs, labels in test_loader:
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        test_embeddings.append(outputs.last_hidden_state.cpu())
        test_labels.append(labels)

train_embeddings = torch.cat(train_embeddings)
train_labels = torch.cat(train_labels)
test_embeddings = torch.cat(test_embeddings)
test_labels = torch.cat(test_labels)


# %%
# save embeddings and labels
torch.save(train_embeddings, "IMDB_train_embeddings.pt")
torch.save(train_labels, "IMDB_train_labels.pt")
torch.save(test_embeddings, "IMDB_test_embeddings.pt")
torch.save(test_labels, "IMDB_test_labels.pt")

# %% [markdown]
# ## Load the IMDB BERT embeddings and labels

# %%
# read embeddings and labels from file
train_embeddings = torch.load("IMDB_train_embeddings.pt")
train_labels = torch.load("IMDB_train_labels.pt")
test_embeddings = torch.load("IMDB_test_embeddings.pt")
test_labels = torch.load("IMDB_test_labels.pt")

# %% [markdown]
# ## Define the classification model and embeddinng dataset

# %%
classifier = torch.nn.Linear(768 * 512, 1).to(DEVICE)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)

random_idx = torch.randperm(len(train_embeddings))

train_dataset = TensorDataset(train_embeddings, train_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# %% [markdown]
# ## Training and evaluation


# %%
# Training function
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for step, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(DEVICE).view(inputs.size(0), -1)
        labels = labels.to(DEVICE).float()
        optimizer.zero_grad()
        # outputs = model(**inputs, labels=labels)
        outputs = model(inputs).squeeze().sigmoid()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


# Evaluation function
def evaluate(model, loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(loader):
            true_labels.extend(labels.tolist())
            inputs = inputs.to(DEVICE).view(inputs.size(0), -1)
            labels = labels.to(DEVICE).float()
            outputs = model(inputs).squeeze() > 0
            preds.extend(outputs.cpu().tolist())
    return accuracy_score(true_labels, preds)


# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    train_loss = train(classifier, train_loader, optimizer)
    test_accuracy = evaluate(classifier, test_loader)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

print("Training complete.")

# %%
