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
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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

class CNNTransformer(ImageClassificationBase):
    """CNN + Transformer混合模型：先用CNN提取特征，再切块加Transformer捕捉全局"""
    def __init__(self, image_size=128, num_classes=38, dim=256, depth=4, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        # 前置 CNN 特征提取: 下采样到 32x32
        self.cnn = nn.Sequential(
            nn.Conv2d(3, dim, 7, stride=2, padding=3),  # 128->64
            nn.BatchNorm2d(dim), nn.ReLU(),
            nn.MaxPool2d(2)  # 64->32
        )
        # Patch embedding
        patch_size = 16
        patch_dim = patch_size * patch_size * dim
        self.to_patches = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        num_patches = (32 // patch_size) ** 2
        # CLS token + 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
                )
            ]) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
    def forward(self, x):
        x = self.cnn(x)
        b = x.shape[0]
        x = self.to_patches(x)  # (b, num_patches, dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for norm1, attn, drop1, norm2, mlp in self.transformer_blocks:
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + drop1(attn_out)
            x_norm2 = norm2(x)
            x = x + mlp(x_norm2)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

class VisionTransformer(ImageClassificationBase):
    """简化版ViT模型：Patch切分 + Transformer Encoder + 分类头"""
    def __init__(self, image_size=128, patch_size=16, num_classes=38, dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        # Patch embedding using separate p1 and p2 to avoid duplicate dimension names
        self.patch_embedding = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size), nn.Linear(patch_dim, dim))
        # CLS token + Position Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        # Transformer Encoder Blocks
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True),
                nn.Dropout(dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embedding(x)  # (b, num_patches, dim)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for norm1, attn, drop1, norm2, mlp in self.transformer_blocks:
            x_norm = norm1(x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + drop1(attn_out)
            x_norm2 = norm2(x)
            x = x + mlp(x_norm2)
        cls_out = x[:, 0]
        return self.mlp_head(cls_out)

class FasterRCNN(ImageClassificationBase):
    """使用Faster R-CNN的骨干网络特征进行分类。仅提取backbone输出，并做全局池化+分类。"""
    def __init__(self, num_classes=38):
        super().__init__()
        # 加载预训练Faster R-CNN，取其backbone (FPN)
        frcnn = fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone = frcnn.backbone  # Feature Pyramid Network backbone
        # 分类头：对任一尺度特征做池化并线性分类
        feat_dim = 256  # FPN最后一层输出通道
        self.classifier = nn.Linear(feat_dim, num_classes)
    def forward(self, images):
        """images: Tensor(batch,3,H,W) 返回logits: Tensor(batch,num_classes)"""
        # backbone 返回 dict of tensors
        feats = self.backbone(images)
        # 取最高分辨率的特征图，比如 feats['0'] 或最后一层 feats list
        # 对每个 batch 做全局池化分类
        # 假设 feats 是 OrderedDict, 取第一个 key
        x = list(feats.values())[0]
        # x: (b, C, H, W)
        pooled = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(pooled)
    
class RepVGGBlock(nn.Module):
    """RepVGG基本结构"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, deploy=False):
        super().__init__()
        self.deploy = deploy
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels==in_channels and stride==1 else None
            self.rbr_dense = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels))
            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False), nn.BatchNorm2d(out_channels))
        self.nonlinearity = nn.ReLU()
    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.nonlinearity(out)
class RepVGG(ImageClassificationBase):
    """RepVGG网络，支持训练时多分支，部署时融合单分支"""
    def __init__(self, num_classes=38, deploy=False):
        super().__init__()
        self.deploy = deploy
        # 简化版RepVGG-A0
        self.stage0 = nn.Sequential(
            RepVGGBlock(3, 64, 3, stride=2, padding=1, deploy=deploy),
            RepVGGBlock(64, 64, 3, stride=1, padding=1, deploy=deploy)
        )
        self.stage1 = nn.Sequential(
            RepVGGBlock(64, 128, 3, stride=2, padding=1, deploy=deploy),
            RepVGGBlock(128, 128, 3, stride=1, padding=1, deploy=deploy)
        )
        self.stage2 = nn.Sequential(
            RepVGGBlock(128, 256, 3, stride=2, padding=1, deploy=deploy),
            RepVGGBlock(256, 256, 3, stride=1, padding=1, deploy=deploy)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


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

def base_study(data_path, results_file='results/base/base_results.json'):
    device = get_default_device()
    print(f"Using device: {device}")
    train_loader, test_loader, classes = prepare_data(data_path)
    train_loader = DeviceDataLoader(train_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)   
    # 定义实验配置
    experiments = {
        'FasterRCNN': FasterRCNN(),
        'CNNTransformer': CNNTransformer(),
        'ViT': VisionTransformer(),
        'RepVGG': RepVGG(deploy=False),
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
    print("BASE STUDY RESULTS SUMMARY")
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
        plt.savefig('results/base/base_study_comparison.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('results/base/base_study_training.png', dpi=300, bbox_inches='tight')
        plt.show()   
    return df

def main():
    data_path = "D:\\本科毕业论文\\Plant\\PlantVillage"
    try:
        results = base_study(data_path)
        comparison_df = analyze_results(results)        
        comparison_df.to_csv('results/base/base_study_summary.csv', index=False)  # 保存详细结果
        print("\nBase study completed!")
        print("Results saved to:")
        print("- base_results.json (detailed results)")
        print("- base_study_summary.csv (summary table)")
        print("- base_study_comparison.png (comparison plots)")
        print("- base_study_training.png (training curves)")        
    except Exception as e:
        print(f"Error during base study: {e}")
        print("Please check your data path and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
