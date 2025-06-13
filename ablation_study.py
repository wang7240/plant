import pandas as pd
import os
import time
from PIL import Image
import random, numpy as np, torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from data_transforms import ToTensorSafe
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, PILToTensor, ConvertImageDtype
import torchvision.models as models
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# 设置随机种子确保实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def prepare_data(data_path, batch_size=32, test_split=0.3):
    transform = ToTensorSafe(size=128)
    dataset = ImageFolder(data_path, transform=transform)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])   
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=False)
    return train_loader, test_loader, dataset.classes

def accuracy(preds, labels):
    return (preds == labels).float().mean()
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss    
    def test_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, labels)
        return {'test_loss': loss.detach(), 'test_acc': acc.detach(), 'preds': preds, 'labels': labels}    
    def test_epoch_end(self, outputs):
        batch_losses = [x['test_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean().item()
        batch_accs = [x['test_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean().item()        
        all_preds = torch.cat([x['preds'] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy()        
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)        
        return { 'test_loss': epoch_loss, 
            'test_acc': epoch_acc, 
            'test_prec': prec, 
            'test_rec': rec, 
            'test_f1': f1 }

# CBAM注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()    
    def forward(self, x):
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        avg_channel = torch.mean(x, dim=1, keepdim=True)
        feat = torch.cat([max_channel, avg_channel], dim=1)
        attn = self.sigmoid(self.conv(feat))
        return attn * x
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

# 消融实验
class BaselineCNN(ImageClassificationBase):
    """基础CNN模型（无任何增强）"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )    
    def forward(self, x):
        return self.network(x)
    
class CNN_WithBatchNorm(ImageClassificationBase):
    """添加批归一化的CNN"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )   
    def forward(self, x):
        return self.network(x)

class CNN_WithDropout(ImageClassificationBase):
    """添加Dropout的CNN"""
    def __init__(self, num_classes=38, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )    
    def forward(self, x):
        return self.network(x)

class CNN_WithChannelAttention(ImageClassificationBase):
    """只添加通道注意力的CNN"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            ChannelAttention(64), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            ChannelAttention(128), nn.MaxPool2d(2) )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            ChannelAttention(256), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes))    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)
class CNN_WithSpatialAttention(ImageClassificationBase):
    """只添加空间注意力的CNN"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            SpatialAttention(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            SpatialAttention(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            SpatialAttention(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, num_classes))   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)
class CNN_WithCBAM(ImageClassificationBase):
    """完整CBAM注意力机制的CNN"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            CBAM(64), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            CBAM(128), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            CBAM(256), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))   
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)
    
class CNN_WithBatchNormCBAM(ImageClassificationBase):
    """添加批归一化 + CBAM 的CNN"""
    def __init__(self, num_classes=38):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            CBAM(64), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            CBAM(128), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CBAM(256), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

class CNN_WithDropoutCBAM(ImageClassificationBase):
    """添加 Dropout + CBAM 的CNN"""
    def __init__(self, num_classes=38, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            CBAM(64), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            CBAM(128), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            CBAM(256), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)

class CNN_WithBatchNormDropout(ImageClassificationBase):
    """添加批归一化 + Dropout 的CNN"""
    def __init__(self, num_classes=38, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.network(x)

class CNN_WithBatchNormDropoutCBAM(ImageClassificationBase):
    """添加批归一化 + Dropout + CBAM 的CNN"""
    def __init__(self, num_classes=38, dropout_rate=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            CBAM(64), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            CBAM(128), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            CBAM(256), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


# GPU设备
def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)   
    def __len__(self):
        return len(self.dl)

# 训练和评估函数
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.test_step(batch) for batch in test_loader]
    return model.test_epoch_end(outputs)
def train_model(model, train_loader, test_loader, epochs=15, lr=0.001, opt_func=torch.optim.Adam):
    """训练单个模型"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()       
        # 评估
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()       
        print(f"Epoch [{epoch+1}], train_loss: {result['train_loss']:.4f}, "
              f"test_loss: {result['test_loss']:.4f}, test_acc: {result['test_acc']:.4f}, "
              f"test_prec: {result['test_prec']:.4f}, test_rec: {result['test_rec']:.4f}, "
              f"test_f1: {result['test_f1']:.4f}")       
        history.append(result)
    training_time = time.time() - start_time
    return history, training_time

# 消融实验
def ablation_study(data_path, results_file='results/ablation/ablation_results.json'):
    device = get_default_device()
    print(f"Using device: {device}")
    train_loader, test_loader, classes = prepare_data(data_path)
    train_loader = DeviceDataLoader(train_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)   
    # 定义实验配置
    experiments = {
        'CNN': BaselineCNN(),
        'CNN-B': CNN_WithBatchNorm(),
        'CNN-D': CNN_WithDropout(),
        'CNN-C': CNN_WithCBAM(),
        'CNN-BC': CNN_WithBatchNormCBAM(),
        'CNN-DC': CNN_WithDropoutCBAM(),
        'CNN-BD': CNN_WithBatchNormDropout(),
        'CNN-BDC': CNN_WithBatchNormDropoutCBAM(),
    }
    results = {}   
    for exp_name, model in experiments.items():
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print('='*50)
        model = to_device(model, device)       
        # 训练模型
        history, training_time = train_model(model, train_loader, test_loader)        
        # 最终评估
        final_results = evaluate(model, test_loader)
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 存储结果
        results[exp_name] = {
            'final_accuracy': final_results['test_acc'],
            'final_precision': final_results['test_prec'],
            'final_recall': final_results['test_rec'],
            'final_f1': final_results['test_f1'],
            'final_loss': final_results['test_loss'],
            'training_time': training_time,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'history': [
                {k: v for k, v in epoch_result.items() if k != 'preds'} 
                for epoch_result in history
            ]
        }       
        print(f"Final Results for {exp_name}:")
        print(f"  Accuracy: {final_results['test_acc']:.4f}")
        print(f"  Precision: {final_results['test_prec']:.4f}")
        print(f"  Recall: {final_results['test_rec']:.4f}")
        print(f"  F1-Score: {final_results['test_f1']:.4f}")
        print(f"  Training Time: {training_time:.2f}s")
        print(f"  Total Parameters: {total_params:,}")
    # 保存结果
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)    
    return results

# 结果分析和可视化
def analyze_results(results, save_plots=True):
    # 创建结果对比表
    comparison_data = []
    for exp_name, exp_results in results.items():
        comparison_data.append({
            'Experiment': exp_name,
            'Accuracy': exp_results['final_accuracy'],
            'Precision': exp_results['final_precision'],
            'Recall': exp_results['final_recall'],
            'F1-Score': exp_results['final_f1'],
            'Loss': exp_results['final_loss'],
        })   
    df = pd.DataFrame(comparison_data)
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))    
    if save_plots:
        # 绘制对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # 准确率对比
        axes[0, 0].bar(df['Experiment'], df['Accuracy'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)        
        # 精确率对比
        axes[0, 1].bar(df['Experiment'], df['Precision'])
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)       
        # 召回率对比
        axes[1, 0].bar(df['Experiment'], df['Recall'])
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45) 
        # F1分数对比
        axes[1, 1].bar(df['Experiment'], df['F1-Score'])
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/ablation/ablation_study_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 训练过程对比
        plt.figure(figsize=(15, 5))        
        plt.subplot(1, 2, 1)
        for exp_name, exp_results in results.items():
            accuracies = [epoch['test_acc'] for epoch in exp_results['history']]
            plt.plot(accuracies, label=exp_name, marker='o', markersize=3)
        plt.title('Test Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)        
        plt.subplot(1, 2, 2)
        for exp_name, exp_results in results.items():
            losses = [epoch['test_loss'] for epoch in exp_results['history']]
            plt.plot(losses, label=exp_name, marker='o', markersize=3)
        plt.title('Test Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)       
        plt.tight_layout()
        plt.savefig('results/ablation/ablation_study_training.png', dpi=300, bbox_inches='tight')
        plt.show()   
    return df

def main():
    data_path = "D:\\本科毕业论文\\Plant\\PlantVillage"
    try:
        results = ablation_study(data_path)
        comparison_df = analyze_results(results)        
        comparison_df.to_csv('results/ablation/ablation_study_summary.csv', index=False)  # 保存详细结果
        print("\nAblation study completed!")
        print("Results saved to:")
        print("- ablation_results.json (detailed results)")
        print("- ablation_study_summary.csv (summary table)")
        print("- ablation_study_comparison.png (comparison plots)")
        print("- ablation_study_training.png (training curves)")        
    except Exception as e:
        print(f"Error during ablation study: {e}")
        print("Please check your data path and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
    