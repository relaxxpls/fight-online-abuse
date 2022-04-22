import torch
import torch.nn as nn


def train(model, dataloader, criterion, optimizer, device="cpu"):
    print("Training...")
    model.train()
    losses, predictions = [], []

    # ? iterate over batches
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)

        # ? Compute the loss between actual and predicted values
        loss = criterion(logits, labels)

        # ? Calculate the gradients via backpropagation
        loss.backward()
        # ? clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # ? Update the model parameters
        optimizer.step()

        labels_pred = torch.sigmoid(logits.detach()).round()
        losses.append(loss.item())
        predictions.append(labels_pred)

        # ? progress update after every 50 batches.
        if (idx + 1) % 50 == 0:
            print(f"  [Batch {idx+1}\t/{len(dataloader)}] Loss: {loss.item():.4f}")

    predictions = torch.cat(predictions, dim=0).cpu()
    losses = torch.tensor(losses).cpu()

    return losses, predictions


def evaluate(model, dataloader, criterion, device="cpu"):
    print("Evaluating...")
    model.eval()
    losses, predictions = [], []

    for idx, batch in enumerate(dataloader):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            logits = model(ids, mask)
            loss = criterion(logits, labels)

        labels_pred = torch.sigmoid(logits.detach()).round()
        losses.append(loss.item())
        predictions.append(labels_pred)

        # ? progress update after every 50 batches.
        if (idx + 1) % 50 == 0:
            print(f"  [Batch {idx+1}\t/{len(dataloader)}] Loss: {loss.item():.4f}")

    predictions = torch.cat(predictions, dim=0).cpu()
    losses = torch.tensor(losses).cpu()

    return losses, predictions
