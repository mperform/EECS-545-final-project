import torch.optim as optim
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from tqdm import tqdm
from isic_metric import score
from sklearn.metrics import roc_auc_score, roc_curve

class Trainer:
    def __init__(self, device, train_dataset, val_dataset, nn_name, weights, transform, model, lr=1e-5, num_epochs=20):
        self.device = device
        self.weights = weights
        self.transform = transform
        self.model = model
        if nn_name == "EfficientNet-B3": # EfficientNet
            print("EfficientNet Configuration")
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, 1)
        elif nn_name == "DenseNet121": # DenseNet
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, 1)
        elif nn_name == "ResNet50": # ResNet
            model.fc = nn.Linear(model.fc.in_features, 1)
        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.partial_aucs = []
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=1)
        self.nn_name = nn_name
        
        # make directory for saving checkpoints
        if not os.path.exists(f"{self.nn_name}_checkpoints"):
            os.makedirs(f"{self.nn_name}_checkpoints")
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                inputs = inputs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)  # [B, 1]

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.train_losses.append(epoch_loss / len(self.train_loader))

            # === Validation ===
            self.model.eval()
            val_loss = 0
            preds, targets = [], []

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    targets.extend(labels.cpu().numpy())

            self.val_losses.append(val_loss / len(self.val_loader))
            val_auc = roc_auc_score(targets, preds)
            
            # Threshold predictions at 0.5
            pred_labels = (np.array(preds) >= 0.5).astype(int)
            true_labels = np.array(targets).astype(int)

            
            # Partial AUC Computation 
            true_labels = np.array(targets).astype(int).flatten()
            preds = np.array(preds).flatten()
            df_sol = pd.DataFrame({
                "image_id": list(range(len(true_labels))),
                "target": true_labels
            })
            
            df_sub = pd.DataFrame({
                "image_id": list(range(len(preds))),
                "target": preds  # prediction probabilities
            })
            
            # Compute pAUC from official metric
            pauc = score(df_sol, df_sub, row_id_column_name="image_id", min_tpr=0.80)
            self.partial_aucs.append(pauc)
            
            val_acc = accuracy_score(true_labels, pred_labels)
            val_precision = precision_score(true_labels, pred_labels)
            val_recall = recall_score(true_labels, pred_labels)
            val_f1 = f1_score(true_labels, pred_labels)

            print(f"Epoch {epoch+1}: Acc = {val_acc:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, F1 = {val_f1:.4f}, pAUC = {pauc:.4f}")    
            self.val_accuracies.append(val_acc)
            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_auc': val_auc
            }, f"{self.nn_name}_checkpoints/{self.nn_name}_epoch_{epoch+1}.pth")
            
        # plot loss curves
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.nn_name} Training Loss")
        plt.legend()
        plt.savefig(f"{self.nn_name}_loss_curve.png")
        
        # plot pAUC
        epochs = list(range(1, self.num_epochs + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, self.partial_aucs, marker='o', label="Partial AUC @ TPR>0.8")
        plt.title(f"{self.nn_name} Partial AUC over Epochs (FPR ≤ 0.2)")
        plt.xlabel("Epoch")
        plt.ylabel("Partial AUC (scaled)")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.nn_name}_partial_auc_curve.png")