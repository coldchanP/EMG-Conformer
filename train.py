"""
EMG Conformer Large Model - Final Version with 5-Fold CV and Visualization

High-capacity 4-channel EMG signal classification with:
- 5-fold cross validation
- Best model weight saving  
- Comprehensive visualization (accuracy plots, confusion matrix, t-SNE, training curves)
- Results organized in timestamped run folders

Enhanced model with:
- Larger embedding size (40 -> 128)
- Deeper transformer (6 -> 10 layers)
- More attention heads (5 -> 16)
- Advanced visualization capabilities
"""

import argparse
import os
import pandas as pd
import glob
gpus = [0]  # GPU configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import itertools
import datetime
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import seaborn as sns

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from torchsummary import summary
try:
    from fvcore.nn import flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not available, FLOPs calculation will be skipped")

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Required for background execution
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=128):  # Increased from 40 to 128
        super().__init__()
        
        # Deep ConvNet for stronger feature extraction
        self.shallownet = nn.Sequential(
            # First convolution block
            nn.Conv2d(1, 64, (1, 7), (1, 1)),   # 40 -> 64 filters
            nn.Conv2d(64, 64, (4, 1), (1, 1)),  # 4-channel spatial convolution
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 3), (1, 2)),
            nn.Dropout(0.3),
            
            # Second convolution block (additional)
            nn.Conv2d(64, 96, (1, 3), (1, 1)),
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.AvgPool2d((1, 3), (1, 2)),
            nn.Dropout(0.4),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(96, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=16,  # Increased from 5 to 16
                 drop_p=0.3,   # Decreased from 0.5 to 0.3 (preserve more information)
                 forward_expansion=6,  # Increased from 4 to 6
                 forward_drop_p=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # Deeper and wider classification head
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256),  # Increased from 64 to 256
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),       # Additional layer
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        # Changed from 32 to 64
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # Combination of Global Average Pooling + Global Max Pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        x = avg_pool + max_pool  # Feature fusion
        
        out = self.fc(x)
        return x, out


class ViT(nn.Sequential):
    def __init__(self, emb_size=128, depth=10, n_classes=4, **kwargs):  # depth 6->10, emb_size 40->128
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class EMGDataset(Dataset):
    def __init__(self, data, labels, window_size=200, overlap=0.5):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.overlap = overlap
        
        # Window generation
        self.windows = []
        self.window_labels = []
        
        step_size = int(window_size * (1 - overlap))
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i+window_size]
            label = labels[i + window_size // 2]  # Use label from center of window
            self.windows.append(window)
            self.window_labels.append(label)
        
        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.window_labels[idx]])


class EMGVisualization:
    """EMG model result visualization class"""
    
    def __init__(self, run_dir, class_names):
        self.run_dir = run_dir
        self.class_names = class_names
        
        # Visualization directory creation
        self.vis_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def plot_training_curves(self, train_history, val_history, fold_results):
        """Training curve visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall training loss curve
        axes[0, 0].plot(train_history['epoch'], train_history['loss'], 'b-', label='Training Loss', alpha=0.7)
        axes[0, 0].plot(val_history['epoch'], val_history['loss'], 'r-', label='Validation Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Overall training accuracy curve
        axes[0, 1].plot(train_history['epoch'], train_history['accuracy'], 'b-', label='Training Accuracy', alpha=0.7)
        axes[0, 1].plot(val_history['epoch'], val_history['accuracy'], 'r-', label='Validation Accuracy', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Best performance per fold
        fold_nums = list(range(1, len(fold_results) + 1))
        best_accs = [result['best_acc'] for result in fold_results]
        avg_accs = [result['avg_acc'] for result in fold_results]
        
        axes[1, 0].bar([x - 0.2 for x in fold_nums], best_accs, 0.4, label='Best Accuracy', alpha=0.7)
        axes[1, 0].bar([x + 0.2 for x in fold_nums], avg_accs, 0.4, label='Average Accuracy', alpha=0.7)
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('5-Fold Cross Validation Results')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Learning Rate scheduling
        if 'lr' in val_history:
            axes[1, 1].plot(val_history['epoch'], val_history['lr'], 'g-', alpha=0.7)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, y_true, y_pred, fold_idx, accuracy):
        """Confusion Matrix visualization"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - Fold {fold_idx+1} (Acc: {accuracy:.2f}%)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'confusion_matrix_fold{fold_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_tsne(self, features, labels, fold_idx, perplexity=30):
        """t-SNE visualization"""
        print(f"t-SNE visualization generation in progress... (Fold {fold_idx+1})")
        
        # Feature vector dimension reduction (if necessary)
        if features.shape[1] > 50:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
        
        # t-SNE application
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        # Normalization
        x_min, x_max = features_2d.min(0), features_2d.max(0)
        features_norm = (features_2d - x_min) / (x_max - x_min)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, class_name in enumerate(self.class_names):
            mask = labels == i
            if np.any(mask):
                plt.scatter(features_norm[mask, 0], features_norm[mask, 1], 
                           c=colors[i], label=class_name, alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title(f't-SNE Visualization - Fold {fold_idx+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'tsne_fold{fold_idx+1}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_class_distribution(self, all_labels):
        """Class distribution visualization"""
        unique, counts = np.unique(all_labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(unique)), counts, color=['red', 'blue', 'green', 'orange'], alpha=0.7)
        
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(len(unique)), [self.class_names[i] for i in unique])
        
        # Display numbers on each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, 'class_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_classification_report(self, y_true, y_pred, fold_idx):
        """Classification report saving"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, 
                                     output_dict=True)
        
        # Saving as text format
        with open(os.path.join(self.vis_dir, f'classification_report_fold{fold_idx+1}.txt'), 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return report


class EMGConformerLargeFinal():
    def __init__(self, n_classes=4, window_size=30):
        super(EMGConformerLargeFinal, self).__init__()
        self.batch_size = 32
        self.n_epochs = 50
        self.img_height = 4
        self.img_width = window_size
        self.channels = 1
        self.c_dim = n_classes
        self.lr = 0.0005
        self.b1 = 0.9
        self.b2 = 0.999
        self.window_size = window_size
        self.n_classes = n_classes
        
        self.data_root = './custom_dataset/data/'
        
        # Timestamp-based execution directory creation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"./runs/emg_conformer_large_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        
        # Class name definition
        self.class_names = ["Rest", "Pinch Grasp", "Cylinder Grasp", "Extenstion"]
        
        # Visualizer object creation
        self.visualizer = EMGVisualization(self.run_dir, self.class_names)
        
        # Log file
        self.log_write = open(os.path.join(self.run_dir, "training_log.txt"), "w")
        
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            self.criterion_cls = self.criterion_cls.cuda()
        
        # Model creation
        self.model = ViT(emb_size=128, depth=10, n_classes=n_classes)
        
        print(f"\n{'='*60}")
        print("EMG CONFORMER LARGE MODEL - FINAL VERSION")
        print(f"Run Directory: {self.run_dir}")
        print(f"{'='*60}")
        
        # Parameter count calculation
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        print(f"{'='*60}\n")

    def save_model(self, fold, epoch, accuracy, is_best=False, is_global_best=False):
        """Model weight saving"""
        # Model state_dict saving (considering DataParallel)
        if hasattr(self.model, 'module'):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        # Saving information
        save_dict = {
            'model_state_dict': model_state,
            'fold': fold,
            'epoch': epoch,
            'accuracy': accuracy,
            'model_config': {
                'emb_size': 128,
                'depth': 10,
                'n_classes': self.n_classes,
                'window_size': self.window_size
            },
            'training_config': {
                'lr': self.lr,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs
            }
        }
        
        checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        
        # File name creation
        if is_global_best:
            filename = f'emg_conformer_large_GLOBAL_BEST_acc{accuracy:.2f}.pth'
        elif is_best:
            filename = f'emg_conformer_large_best_fold{fold+1}_acc{accuracy:.2f}.pth'
        else:
            filename = f'emg_conformer_large_fold{fold+1}_epoch{epoch}_acc{accuracy:.2f}.pth'
        
        filepath = os.path.join(checkpoint_dir, filename)
        
        # Weight saving
        torch.save(save_dict, filepath)
        print(f"Model saved: {filepath}")
        
        return filepath

    def load_data(self):
        """Loading EMG data from all CSV files"""
        all_data = []
        all_labels = []
        
        csv_files = glob.glob(self.data_root + "*.csv")
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            print(f"Loading {csv_file}")
            df = pd.read_csv(csv_file, header=None)
            
            # First 4 columns are EMG signals, last column is label
            data = df.iloc[:, :4].values.astype(np.float32)  # EMG signals
            labels = df.iloc[:, 4].values.astype(np.int64)   # Labels
            
            # Normalization (0-255 -> 0-1)
            data = data / 255.0
            
            all_data.append(data)
            all_labels.append(labels)
        
        # Data combination from all files
        self.all_data = np.concatenate(all_data, axis=0)
        self.all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"Total data shape: {self.all_data.shape}")
        print(f"Total labels shape: {self.all_labels.shape}")
        print(f"Label distribution: {np.bincount(self.all_labels)}")
        
        # Class distribution visualization
        self.visualizer.plot_class_distribution(self.all_labels)
        
        return self.all_data, self.all_labels

    def get_cross_validation_data(self, fold, n_folds=5):
        """Data splitting for K-fold cross validation"""
        data, labels = self.load_data()
        
        # Data splitting into n_folds
        fold_size = len(data) // n_folds
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(data)
        
        # Test set
        test_data = data[start_idx:end_idx]
        test_labels = labels[start_idx:end_idx]
        
        # Training set
        train_data = np.concatenate([data[:start_idx], data[end_idx:]], axis=0)
        train_labels = np.concatenate([labels[:start_idx], labels[end_idx:]], axis=0)
        
        # Normalization (based on training data)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        print(f"Fold {fold+1}: Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        
        return train_data, train_labels, test_data, test_labels

    def train_single_fold(self, fold):
        """Single fold training"""
        train_data, train_labels, test_data, test_labels = self.get_cross_validation_data(fold)
        
        # Model reinitialization (new model for each fold)
        self.model = ViT(emb_size=128, depth=10, n_classes=self.n_classes)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
            self.model = self.model.cuda()
        
        # Dataset creation
        train_dataset = EMGDataset(train_data, train_labels, window_size=self.window_size)
        test_dataset = EMGDataset(test_data, test_labels, window_size=self.window_size)
        
        print(f"Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)}")
        
        # DataLoader creation
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, 
                                     betas=(self.b1, self.b2), weight_decay=0.01)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)
        
        # Training record variable
        train_history = {'epoch': [], 'loss': [], 'accuracy': []}
        val_history = {'epoch': [], 'loss': [], 'accuracy': [], 'lr': []}
        
        best_acc = 0
        total_acc = 0
        num_evals = 0
        
        # Variable for collecting feature vectors and labels
        all_features = []
        all_labels_pred = []
        all_labels_true = []
        
        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data = Variable(data.type(self.Tensor))
                target = Variable(target.squeeze().type(self.LongTensor))
                
                optimizer.zero_grad()
                features, outputs = self.model(data)
                loss = self.criterion_cls(outputs, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            
            # Learning rate update
            scheduler.step()
            
            # Evaluation (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0
                
                # Collect feature vectors and predictions at the last epoch
                if epoch >= self.n_epochs - 10:  # Last 10 epochs
                    epoch_features = []
                    epoch_pred = []
                    epoch_true = []
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data = Variable(data.type(self.Tensor))
                        target = Variable(target.squeeze().type(self.LongTensor))
                        
                        features, outputs = self.model(data)
                        test_loss += self.criterion_cls(outputs, target).item()
                        pred = outputs.argmax(dim=1)
                        test_correct += pred.eq(target).sum().item()
                        test_total += target.size(0)
                        
                        # Collect feature vectors at the last epoch
                        if epoch >= self.n_epochs - 10:
                            epoch_features.append(features.cpu().numpy())
                            epoch_pred.append(pred.cpu().numpy())
                            epoch_true.append(target.cpu().numpy())
                
                train_acc = 100. * train_correct / train_total
                test_acc = 100. * test_correct / test_total
                current_lr = scheduler.get_last_lr()[0]
                
                print(f'Fold {fold+1} - Epoch: {epoch+1:3d} | LR: {current_lr:.6f} | '
                      f'Train Loss: {train_loss/len(train_loader):.4f} | '
                      f'Train Acc: {train_acc:.2f}% | Test Loss: {test_loss/len(test_loader):.4f} | '
                      f'Test Acc: {test_acc:.2f}%')
                
                # Training record saving
                train_history['epoch'].append(epoch + 1)
                train_history['loss'].append(train_loss / len(train_loader))
                train_history['accuracy'].append(train_acc)
                
                val_history['epoch'].append(epoch + 1)
                val_history['loss'].append(test_loss / len(test_loader))
                val_history['accuracy'].append(test_acc)
                val_history['lr'].append(current_lr)
                
                self.log_write.write(f"Fold {fold+1}\t{epoch+1}\t{test_acc:.4f}\t{current_lr:.6f}\n")
                
                total_acc += test_acc
                num_evals += 1
                
                # Update best performance if test_acc > best_acc
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.save_model(fold, epoch+1, test_acc, is_best=True)
                
                # Collect feature vectors at the last epoch
                if epoch >= self.n_epochs - 10:
                    all_features = np.concatenate(epoch_features, axis=0)
                    all_labels_pred = np.concatenate(epoch_pred, axis=0)
                    all_labels_true = np.concatenate(epoch_true, axis=0)
        
        avg_acc = total_acc / num_evals
        print(f'Fold {fold+1} - Best Acc: {best_acc:.2f}%, Average Acc: {avg_acc:.2f}%')
        
        # Visualization creation
        self.visualizer.plot_confusion_matrix(all_labels_true, all_labels_pred, fold, best_acc)
        self.visualizer.plot_tsne(all_features, all_labels_true, fold)
        self.visualizer.save_classification_report(all_labels_true, all_labels_pred, fold)
        
        return {
            'best_acc': best_acc, 
            'avg_acc': avg_acc,
            'train_history': train_history,
            'val_history': val_history,
            'y_true': all_labels_true,
            'y_pred': all_labels_pred
        }

    def train_5fold(self):
        """5-fold cross validation training"""
        n_folds = 5
        fold_results = []
        all_train_history = {'epoch': [], 'loss': [], 'accuracy': []}
        all_val_history = {'epoch': [], 'loss': [], 'accuracy': [], 'lr': []}
        
        print(f"\n{'='*60}")
        print("5-FOLD CROSS VALIDATION START")
        print(f"{'='*60}")
        
        global_best_acc = 0
        global_best_fold = -1
        
        for fold in range(n_folds):
            print(f"\n{'='*50}")
            print(f"FOLD {fold+1}/{n_folds} START")
            print(f"{'='*50}")
            
            fold_result = self.train_single_fold(fold)
            fold_results.append(fold_result)
            
            # Overall best performance tracking
            if fold_result['best_acc'] > global_best_acc:
                global_best_acc = fold_result['best_acc']
                global_best_fold = fold
                # Save overall best performance model
                self.save_model(fold, -1, global_best_acc, is_global_best=True)
            
            # Training record accumulation
            for key in all_train_history.keys():
                all_train_history[key].extend(fold_result['train_history'][key])
            for key in all_val_history.keys():
                all_val_history[key].extend(fold_result['val_history'][key])
        
        # Overall result calculation
        final_best_acc = np.mean([result['best_acc'] for result in fold_results])
        final_avg_acc = np.mean([result['avg_acc'] for result in fold_results])
        
        print(f"\n{'='*60}")
        print("5-FOLD CROSS VALIDATION RESULT:")
        print(f"Average Best Accuracy: {final_best_acc:.2f}% (±{np.std([r['best_acc'] for r in fold_results]):.2f})")
        print(f"Average Mean Accuracy: {final_avg_acc:.2f}% (±{np.std([r['avg_acc'] for r in fold_results]):.2f})")
        print(f"Global Best Accuracy: {global_best_acc:.2f}% (Fold {global_best_fold+1})")
        print(f"{'='*60}")
        
        # Overall visualization creation
        self.visualizer.plot_training_curves(all_train_history, all_val_history, fold_results)
        
        # Result saving
        result_file = os.path.join(self.run_dir, "final_results.txt")
        with open(result_file, "w") as f:
            f.write("5-Fold Cross Validation Results:\n")
            f.write(f"Average Best Accuracy: {final_best_acc:.4f} (±{np.std([r['best_acc'] for r in fold_results]):.4f})\n")
            f.write(f"Average Mean Accuracy: {final_avg_acc:.4f} (±{np.std([r['avg_acc'] for r in fold_results]):.4f})\n")
            f.write(f"Global Best Accuracy: {global_best_acc:.4f} (Fold {global_best_fold+1})\n\n")
            
            for i, result in enumerate(fold_results):
                f.write(f"Fold {i+1}: Best={result['best_acc']:.4f}, Avg={result['avg_acc']:.4f}\n")
        
        return fold_results, global_best_acc, global_best_fold


def main():
    # Result saving directory creation
    os.makedirs("./runs", exist_ok=True)
    
    print("EMG Conformer Large Model - Final Version with 5-Fold CV and Visualization")
    print("="*80)
    
    # seed setting
    seed_n = np.random.randint(2021, 2024)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
    
    print(f"Random seed: {seed_n}")
    
    # Model creation and training
    emg_model = EMGConformerLargeFinal(n_classes=4, window_size=30)
    fold_results, global_best_acc, global_best_fold = emg_model.train_5fold()
    
    print(f"Training completed!")
    print(f"Result saving location: {emg_model.run_dir}")
    print(f"Best performance: {global_best_acc:.2f}% (Fold {global_best_fold+1})")
    print(f"Weight file: {emg_model.run_dir}/checkpoints/")
    print(f"Visualization file: {emg_model.run_dir}/visualizations/")


if __name__ == "__main__":
    print(f"Start time: {time.asctime(time.localtime(time.time()))}")
    main()
    print(f"End time: {time.asctime(time.localtime(time.time()))}") 