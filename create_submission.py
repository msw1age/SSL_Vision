import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import timm
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import randint, loguniform

warnings.filterwarnings('ignore') 

# ============================================================================
#                           MODEL SECTION
# ============================================================================

class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, n_layers=3, use_bn=False):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn: layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOFeatureExtractor:
    def __init__(self, checkpoint_path, device='cuda', use_teacher=True, use_tta=False, concat_features=False):
        self.device = device
        self.use_teacher = use_teacher
        self.use_tta = use_tta
        self.concat_features = concat_features
        
        print(f"Loading DINO model: {checkpoint_path}")
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0, img_size=96, dynamic_img_size=True)
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if use_teacher and 'teacher_backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['teacher_backbone'], strict=False)
        elif 'student_backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['student_backbone'], strict=False)
        
        self.backbone.eval()
        self.backbone = self.backbone.to(device)
        self.embed_dim = self.backbone.embed_dim
        
        if use_tta: print("  TTA Enabled")
        
        self.transform = T.Compose([
            T.Resize((96, 96), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.tta_transforms = [
            T.Compose([T.Resize((96, 96), interpolation=InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((96, 96), interpolation=InterpolationMode.BILINEAR), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.CenterCrop(96), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.CenterCrop(96), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.Lambda(lambda img: T.functional.crop(img, 0, 0, 96, 96)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.Lambda(lambda img: T.functional.crop(img, 0, 14, 96, 96)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.Lambda(lambda img: T.functional.crop(img, 14, 0, 96, 96)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((110, 110), interpolation=InterpolationMode.BILINEAR), T.Lambda(lambda img: T.functional.crop(img, 14, 14, 96, 96)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        ]

    def _extract_from_tensor(self, img_tensors):
        with torch.no_grad():
            if self.concat_features:
                x = self.backbone.forward_features(img_tensors)
                cls_token = x[:, 0]
                patch_mean = x[:, 1:].mean(dim=1)
                features = torch.cat([cls_token, patch_mean], dim=1)
            else:
                features = self.backbone(img_tensors)
        return features

    def extract_batch_features(self, images):
        if self.use_tta:
            return self._extract_batch_features_tta(images)
        img_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        features = self._extract_from_tensor(img_tensors)
        features = F.normalize(features, dim=1, p=2)
        return features.cpu().numpy()

    def _extract_batch_features_tta(self, images):
        all_features = []
        for transform in self.tta_transforms:
            img_tensors = torch.stack([transform(img) for img in images]).to(self.device)
            features = self._extract_from_tensor(img_tensors)
            features = F.normalize(features, dim=1, p=2)
            all_features.append(features)
        avg_features = torch.stack(all_features).mean(dim=0)
        return F.normalize(avg_features, dim=1, p=2).cpu().numpy()

# ============================================================================
#                           DATA SECTION
# ============================================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_list, labels=None, resolution=96):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(self.image_dir / img_name).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name

def collate_fn(batch):
    if len(batch[0]) == 3: 
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames

def extract_features_from_dataloader(feature_extractor, dataloader, split_name='train'):
    all_features = []
    all_labels = []
    all_filenames = []
    print(f"\nExtracting features from {split_name} set...")
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if len(batch) == 3:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch
        features = feature_extractor.extract_batch_features(images)
        all_features.append(features)
        all_filenames.extend(filenames)
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    return features, labels, all_filenames

# ============================================================================
#                           LINEAR TRAINING
# ============================================================================

def optimize_and_train_linear(X_train, y_train, X_val, y_val, n_iter=20):
    print(f"\n{'='*10} Optimizing Linear Evaluator {'='*10}")
    
    X_combined = np.vstack((X_train, X_val))
    y_combined = np.concatenate((y_train, y_val))
    test_fold = np.concatenate([np.full(len(y_train), -1), np.full(len(y_val), 0)])
    ps = PredefinedSplit(test_fold)
    
    pipeline = Pipeline([
        ('pca', PCA(whiten=True)), 
        ('lr', LogisticRegression(solver='lbfgs', max_iter=2000, n_jobs=-1))
    ])
    
    max_pca = min(X_train.shape[1], X_train.shape[0])
    params = {
        'pca__n_components': randint(50, min(300, max_pca)), 
        'lr__C': loguniform(1e-2, 1e2),
    }
    
    print(f"Searching {n_iter} hyperparameter combinations (Train -> Val)...")
    
    search = RandomizedSearchCV(
        pipeline, params, n_iter=n_iter, cv=ps, 
        scoring='accuracy', verbose=1, n_jobs=-1, random_state=42,
        refit=False 
    )
    search.fit(X_combined, y_combined)
    
    print(f"\nBest Validation Score: {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    
    print("\nRetraining final model")
    final_pipeline = Pipeline([
        ('pca', PCA(n_components=search.best_params_['pca__n_components'], whiten=True)), 
        ('lr', LogisticRegression(C=search.best_params_['lr__C'], solver='lbfgs', max_iter=2000, n_jobs=-1))
    ])
    
    final_pipeline.fit(X_train, y_train)
    
    val_acc = accuracy_score(y_val, final_pipeline.predict(X_val))
    print(f"Sanity Check - Final Model Val Accuracy: {val_acc:.4f}")
    
    return final_pipeline

def create_submission(classifier, X_test, test_filenames, output_path):
    print("\nGenerating predictions on test set...")
    predictions = classifier.predict(X_test)
    submission_df = pd.DataFrame({'id': test_filenames, 'class_id': predictions})
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

# ============================================================================
#                           MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Strict Linear Evaluator')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='submission_linear_strict.csv')
    parser.add_argument('--resolution', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--tta', action='store_true', help='Enable TTA')
    parser.add_argument('--concat', action='store_true', help='Concat CLS + Patch Mean')
    parser.add_argument('--n_iter', type=int, default=20) 
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    data_dir = Path(args.data_dir)

    print("\nLoading metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_df = pd.read_csv(data_dir / 'test_images.csv')

    print(f"\nCreating datasets (res={args.resolution})...")
    train_dataset = ImageDataset(data_dir / 'train', train_df['filename'].tolist(), train_df['class_id'].tolist(), args.resolution)
    val_dataset = ImageDataset(data_dir / 'val', val_df['filename'].tolist(), val_df['class_id'].tolist(), args.resolution)
    test_dataset = ImageDataset(data_dir / 'test', test_df['filename'].tolist(), labels=None, resolution=args.resolution)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    feature_extractor = DINOFeatureExtractor(args.checkpoint, device=device, use_tta=args.tta, concat_features=args.concat)
    
    train_features, train_labels, _ = extract_features_from_dataloader(feature_extractor, train_loader, 'train')
    val_features, val_labels, _ = extract_features_from_dataloader(feature_extractor, val_loader, 'val')
    test_features, _, test_filenames = extract_features_from_dataloader(feature_extractor, test_loader, 'test')

    best_model = optimize_and_train_linear(
        train_features, train_labels, 
        val_features, val_labels, 
        n_iter=args.n_iter
    )

    create_submission(best_model, test_features, test_filenames, args.output)

if __name__ == "__main__":
    main()
