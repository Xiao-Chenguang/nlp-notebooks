import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS
from transformers import (
    AdamW,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

# Load AG News Dataset
train_datapip = AG_NEWS(split="train")  # type: ignore
test_datapip = AG_NEWS(split="test")  # type: ignore

# Define tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=4)


# Preprocessing and Tokenization function
def preprocess(batch):
    texts, labels = zip(*batch)
    inputs = tokenizer(
        list(texts), padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    labels = torch.tensor(labels) - 1  # Label 0-indexed for PyTorch
    return inputs, labels


# DataLoader creation
batch_size = 8

train_loader = DataLoader(
    train_datapip, batch_size=batch_size, shuffle=True, collate_fn=preprocess
)
test_loader = DataLoader(test_datapip, batch_size=batch_size, collate_fn=preprocess)

# Define Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
)


# Training function
def train(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Evaluation function
def evaluate(model, loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, preds)


# Training loop
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, scheduler)
    test_accuracy = evaluate(model, test_loader)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
    )

print("Training complete.")
