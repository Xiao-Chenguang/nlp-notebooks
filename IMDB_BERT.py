# %%
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import IMDB
from transformers import (
    AdamW,
    BertModel,
    BertTokenizer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# ## Prepare the IMDB dataset

# %%
# Load AG News Dataset
train_datapip = IMDB(split="train")  # type: ignore
test_datapip = IMDB(split="test")  # type: ignore

# Define tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)  # type: ignore


# Preprocessing and Tokenization function
def preprocess(batch):
    labels, texts = zip(*batch)
    inputs = tokenizer(
        list(texts), padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    labels = torch.tensor(labels) - 1  # Label 0-indexed for PyTorch
    return inputs, labels


# DataLoader creation
batch_size = 32

train_loader = DataLoader(
    train_datapip, shuffle=True, batch_size=batch_size, collate_fn=preprocess
)
test_loader = DataLoader(test_datapip, batch_size=batch_size, collate_fn=preprocess)

# %% [markdown]
# ## Convert text input into BERT embeddings

# %%
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []
with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        train_embeddings.append(outputs.last_hidden_state.cpu())
        train_labels.append(labels)
    for inputs, labels in test_loader:
        outputs = model(**{k: v.to(DEVICE) for k, v in inputs.items()})
        test_embeddings.append(outputs.last_hidden_state.cpu())
        test_labels.append(labels)

train_embeddings = torch.cat(train_embeddings)
train_labels = torch.cat(train_labels)
test_embeddings = torch.cat(test_embeddings)
test_labels = torch.cat(test_labels)

# save embeddings and labels
torch.save(train_embeddings, "IMDB_train_embeddings.pt")
torch.save(train_labels, "IMDB_train_labels.pt")
torch.save(test_embeddings, "IMDB_test_embeddings.pt")
torch.save(test_labels, "IMDB_test_labels.pt")

# %% [markdown]
# ## Define the classification model

# %%
classifier = torch.nn.Linear(768, 2).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(classifier.parameters(), lr=1e-5)

train_dataset = TensorDataset(train_embeddings, train_labels)
test_dataset = TensorDataset(test_embeddings, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


# Training function
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for step, (inputs, labels) in enumerate(loader):
        optimizer.zero_grad()
        # outputs = model(**inputs, labels=labels)
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
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
            outputs = model(inputs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, preds)


# Training loop
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)  # type: ignore

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer)
    test_accuracy = evaluate(model, test_loader)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

print("Training complete.")
