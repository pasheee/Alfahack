import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from collections import deque
from sklearn.preprocessing import StandardScaler
from copy import deepcopy

class ROCStarLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(ROCStarLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]

        pos_expanded = pos.view(-1, 1)
        neg_expanded = neg.view(1, -1)

        pairwise_diff = neg_expanded - pos_expanded
        pairwise_exp = torch.exp(self.gamma * pairwise_diff)

        loss = torch.mean(pairwise_exp)
        return loss

def save_to_csv(y_pred, ids):
    df = pd.DataFrame({'target': y_pred}).assign(id=ids)
    print(df['target'].value_counts())
    df.to_csv('y_pred.csv', index=False)


def test_model(model, X, scaler, device = torch.device('cuda')):
    model.eval()
    X = X.to_numpy()
    X = scaler.transform(X)
    X = torch.DoubleTensor(X)
    
    with torch.no_grad():
        inputs = X.to(device)
        
        outputs = model(inputs).squeeze()
        
        probabilities = torch.sigmoid(outputs.cpu())
        probabilities = probabilities.numpy()

    y_pred = probabilities
    return y_pred
    
def train_test_model(model, X, y, X_test = None, batch_size=15000, epochs=100, learning_rate=0.001, device=torch.device('cuda'), resample = True, test_size = 0.2, roc = True):

    def calculate_roc_auc(model, X_val, y_test, device=torch.device('cuda')):
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            inputs = X_val.to(device)
            
            outputs = model(inputs).squeeze()
            
            probabilities = torch.sigmoid(outputs.cpu())
            probabilities = probabilities.numpy()
            all_predictions.extend(probabilities)
            all_targets.extend(y_test.cpu().numpy())
                
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
    
        roc_auc = roc_auc_score(all_targets, all_predictions)
        
        return roc_auc

    
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state = 52
    )
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    
    if resample == True:
        ros = RandomOverSampler(random_state=52)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
    # X_val = torch.FloatTensor(X_val)
    # y_val = torch.FloatTensor(y_val)
    
    # train_dataset = TensorDataset(
    #     torch.FloatTensor(X_train),
    #     torch.FloatTensor(y_train)
    # )
    # val_dataset = TensorDataset(
    #     torch.FloatTensor(X_val),
    #     torch.FloatTensor(y_val)
    # )

    X_val = torch.DoubleTensor(X_val)
    y_val = torch.DoubleTensor(y_val)
    
    train_dataset = TensorDataset(
        torch.DoubleTensor(X_train),
        torch.DoubleTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.DoubleTensor(X_val),
        torch.DoubleTensor(y_val)
    )
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    if roc:
        criterion = ROCStarLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    smooth_window = 20
    train_losses = deque(maxlen=smooth_window)
    val_losses = []
    best_val_loss = float(0)
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X.to(device)).squeeze()
            loss = criterion(outputs, batch_y.to(device))
            
            # Backward pass с градиентным клиппированием
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Сглаживание loss    loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        avg_train_loss = np.mean(epoch_losses)
        train_losses.append(avg_train_loss)
        smooth_loss = np.mean(list(train_losses))
        
        # Валидация
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_outputs = model(batch_X.to(device)).squeeze()
                val_loss = criterion(val_outputs, batch_y.to(device))
                val_epoch_losses.append(val_loss.item())
                
        avg_val_loss = np.mean(val_epoch_losses)
        val_losses.append(avg_val_loss)
        avg_auc_val_score = calculate_roc_auc(model, X_val, y_val)

        
        if avg_auc_val_score > best_val_loss:
            best_val_loss = avg_auc_val_score
            best_model = deepcopy(model.state_dict())
            
        scheduler.step(avg_val_loss)
        
        if epoch % 1 == 0:
            print(f'Epoch {epoch}: Train Loss = {smooth_loss:.6f}, Val Loss = {avg_val_loss:.6f}, AUC Score = {avg_auc_val_score:.6f}')
            
    model.load_state_dict(best_model)

    def calculate_roc_auc_wp(model, X_val, y_test, device=torch.device('cuda')):
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            inputs = X_val.to(device)
            
            outputs = model(inputs).squeeze()
            
            probabilities = torch.sigmoid(outputs.cpu())
            probabilities = probabilities.numpy()
            all_predictions.extend(probabilities)
            all_targets.extend(y_test.cpu().numpy())
                
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
    
        roc_auc = roc_auc_score(all_targets, all_predictions)
    
        fpr, tpr, thresholds = roc_curve(all_targets, all_predictions)
    
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        return roc_auc

    
    
    print(f'Best model`s AUC score = {calculate_roc_auc_wp(model, X_val, y_val)}')
    
    if X_test is not None:
        y_pred = test_model(model, X_test.drop(['id'], axis = 1), scaler, device = device)
        ids = X_test['id']     
        save_to_csv(y_pred, ids)
        
    return model, train_losses, val_losses
    