# Sections of this code were inspired and modified by the following tutorial:
# https://www.geeksforgeeks.org/deep-learning/building-a-convolutional-neural-network-using-pytorch/
# These sections are noted by GFG comment above the relevant code blocks

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# AI Assistance: ChatGPT 5.4 was used to help write the importing and processing of the satellite image dataset
# Date: 4/5/2026
# Modifications: Added RGB conversion for color images
class SatelliteDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.images = []
        self.labels = labels

        for path in filepaths:
            img = Image.open(path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.tensor(img).permute(2, 0, 1)
            self.images.append(img)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx], dtype=torch.long)


# inspired by GFG's section 4
class BaselineCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # dynamically determine the flattened feature size based on the input image size 
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# inspired by GFG's section 5, but modified to return prediction metrics
def evaluate(model, loader, device):
    model.eval()
    preds = []
    truths = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            truths.extend(y.cpu().numpy())

    acc = accuracy_score(truths, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truths, preds, average='macro'
    )

    return acc, precision, recall, f1, truths, preds


def get_files(root_dir):
    filepaths = []
    labels = []
    class_names = sorted(os.listdir(root_dir))

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith(".tif"):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(label_idx)

    return filepaths, labels, class_names


def main():
    data_directory = r"C:\Users\gregc\Downloads\128\output"
    img_size = 128 # 28 / 128 / 224
    epochs = 25
    batch_size = 64
    learning_rate = 1e-3
    model_save_path = "best_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    filepaths, labels, class_names = get_files(data_directory)

    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1/3,
        stratify=y_temp,
        random_state=42
    )

    train_loader = DataLoader(
        SatelliteDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        SatelliteDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        SatelliteDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = BaselineCNN(img_size, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # AI Assistance: ChatGPT 5.4 was used to help write the saving model checkpoint code based on validation accuracy
        # Date: 4/5/2026
        # Modifications: Added custom model save path
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

    print("\nLoading best model...")
    model.load_state_dict(torch.load(model_save_path))

    test_acc, test_prec, test_rec, test_f1, truths, preds = evaluate(
        model, test_loader, device
    )

    print("\nTEST RESULTS:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")

    cm = confusion_matrix(truths, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()