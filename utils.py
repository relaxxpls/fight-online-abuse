import torch
import torch.nn as nn


def train(model, dataloader, criterion, optimizer, device="cpu"):
    print("Training...")
    model.train()

    total_loss = 0
    total_preds = []

    # ? iterate over batches
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(ids, mask)

        # ? Compute the loss between actual and predicted values
        loss = criterion(outputs, labels)

        # ? Calculate the gradients via backpropagation
        loss.backward()
        # ? clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # ? Update the model parameters
        optimizer.step()

        total_loss = total_loss + loss.item()
        total_preds.append(outputs.detach().cpu().numpy())

        # ? progress update after every 50 batches.
        if (idx + 1) % 50 == 0:
            print(f"  [Batch {idx+1}\t/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    total_preds = total_preds.reshape(-1, total_preds.shape[-1])

    return avg_loss, total_preds


def evaluate(model, dataloader, criterion, device="cpu"):
    print("Evaluating...")
    model.eval()

    total_loss = 0
    total_preds = []

    for idx, batch in enumerate(dataloader):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(ids, mask)

            # ? Compute the loss between actual and predicted values
            loss = criterion(outputs, labels)

            total_loss = total_loss + loss.item()
            total_preds.append(outputs.detach().cpu().numpy())

        # ? progress update after every 50 batches.
        if (idx + 1) % 50 == 0:
            print(f"  [Batch {idx+1}\t/{len(dataloader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    total_preds = total_preds.reshape(-1, total_preds.shape[-1])

    return avg_loss, total_preds
