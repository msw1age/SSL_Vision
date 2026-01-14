import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
import pandas as pd
from PIL import Image
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATA_DIRS = [
    "/dev/shm/ram_data/train",       
    "/dev/shm/ram_data/openimages",
    "/dev/shm/ram_data/inat_birds_96",
    "/dev/shm/ram_data/inat_birds_96",
    "/dev/shm/ram_data/inat_birds_96",
    "/dev/shm/ram_data/inat_birds_96",
    "/dev/shm/ram_data/inat_birds_96",
]

CUB_DIR = "/scratch/mlh9709/testset_1/data"
CHECKPOINT_DIR = "checkpoints_stage3"
CHECKPOINT_PATH = "dino_stage3_latest.pth"

# ============================================================================
#                          1. AUGMENTATIONS 
# ============================================================================
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            sigma = torch.rand(1).item() * 1.9 + 0.1
            return T.GaussianBlur(kernel_size=3, sigma=(sigma, sigma))(img)
        return img

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return TF.solarize(img, threshold=128)
        return img

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, n_local_crops, global_crop_size=96, local_crop_size=64):
        self.n_local_crops = n_local_crops
        
        self.global_transfo1 = T.Compose([
            T.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.global_transfo2 = T.Compose([
            T.RandomResizedCrop(global_crop_size, scale=global_crops_scale, interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            GaussianBlur(p=0.1),
        ])
        
        self.local_transfo = T.Compose([
            T.RandomResizedCrop(local_crop_size, scale=(0.25, 0.5), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.2, 0.2, 0.1, 0), 
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.n_local_crops):
            crops.append(self.local_transfo(image))
        return crops

# ============================================================================
#                          2. MODEL ARCHITECTURE
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

class DinoLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp_schedule, center_momentum=0.9):
        super().__init__()
        self.student_temp = 0.1
        self.teacher_temp_schedule = teacher_temp_schedule
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, epoch):
        teacher_temp = self.teacher_temp_schedule[epoch]
        
        student_out = [F.softmax(s / self.student_temp, dim=-1) for s in student_output]
        teacher_out = [(t - self.center) / teacher_temp for t in teacher_output]
        teacher_out = [F.softmax(t, dim=-1).detach() for t in teacher_out]
        
        total_loss = 0
        n_loss_terms = 0
        for t_idx, t_out in enumerate(teacher_out):
            for s_idx, s_out in enumerate(student_out):
                if t_idx == s_idx: continue
                loss = torch.sum(-t_out * F.log_softmax(student_output[s_idx] / self.student_temp, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        
        self.update_center(teacher_output)
        return total_loss / n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class DINO(nn.Module):
    def __init__(self, backbone, in_dim, out_dim, n_global_crops):
        super().__init__()
        self.n_global_crops = n_global_crops
        self.student_backbone = backbone
        self.student_head = DinoHead(in_dim, out_dim)
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DinoHead(in_dim, out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters(): p.requires_grad = False
        for p in self.teacher_head.parameters(): p.requires_grad = False
            
    def forward(self, images_list):
        teacher_output = []
        global_crops = images_list[:self.n_global_crops]
        with torch.no_grad():
            for img in global_crops:
                cls_token = self.teacher_backbone(img)
                teacher_output.append(self.teacher_head(cls_token))
        student_output = []
        for img in images_list:
            cls_token = self.student_backbone(img)
            student_output.append(self.student_head(cls_token))
        return student_output, teacher_output

    @torch.no_grad()
    def update_teacher_ema(self, momentum):
        for param_s, param_t in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data.mul_(momentum).add_((1 - momentum) * param_s.data)

# ============================================================================
#                          3. DATASETS & EVAL
# ============================================================================
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        if os.path.exists(root):
            print(f"Scanning {root}...")
            for entry in os.scandir(root):
                if entry.is_file() and entry.name.lower().endswith(valid_ext):
                    self.samples.append(entry.path)
            print(f" -> Found {len(self.samples)} images.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        try: sample = Image.open(self.samples[index]).convert('RGB')
        except: sample = Image.new('RGB', (96, 96))
        if self.transform: sample = self.transform(sample)
        return sample, 0

class CUBEvalDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train')
            self.csv_path = os.path.join(root_dir, 'train_labels.csv')
        else:
            self.img_dir = os.path.join(root_dir, 'val')
            self.csv_path = os.path.join(root_dir, 'val_labels.csv')
        if not os.path.exists(self.csv_path): raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        try: image = Image.open(img_path).convert('RGB')
        except: image = Image.new('RGB', (96, 96))
        if self.transform: image = self.transform(image)
        return image, row['class_id']

@torch.no_grad()
def run_knn_evaluation(teacher_backbone, device, k=10):
    print(f"   Starting CUB-200 k-NN evaluation (k={k})...")
    if not os.path.exists(CUB_DIR): return 0.0

    val_transform = T.Compose([
        T.Resize((96, 96), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    try:
        train_ds = CUBEvalDataset(CUB_DIR, split='train', transform=val_transform)
        val_ds = CUBEvalDataset(CUB_DIR, split='val', transform=val_transform)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    except: return 0.0

    def extract_features(loader):
        features_list, labels_list = [], []
        teacher_backbone.eval()
        for img, label in loader:
            img = img.to(device)
            features = teacher_backbone(img) 
            features_list.append(features)
            labels_list.append(label)
        return torch.cat(features_list).to(device), torch.cat(labels_list).to(device)

    train_features, train_labels = extract_features(train_loader)
    train_features = F.normalize(train_features, dim=1, p=2)
    val_features, val_labels = extract_features(val_loader)
    val_features = F.normalize(val_features, dim=1, p=2)

    total_correct = 0
    for idx in range(0, val_features.size(0), 128):
        batch_features = val_features[idx : idx + 128]
        batch_labels = val_labels[idx : idx + 128]
        similarity = torch.mm(batch_features, train_features.T)
        _, top_k_indices = similarity.topk(k, dim=1, largest=True, sorted=True)
        neighbor_labels = train_labels[top_k_indices]
        voted_labels, _ = torch.mode(neighbor_labels, dim=1)
        total_correct += (voted_labels == batch_labels).sum().item()

    accuracy = 100 * (total_correct / len(val_ds))
    print(f"   --- CUB Validation Accuracy (k={k}): {accuracy:.2f}% ---")
    teacher_backbone.train()
    return accuracy

# ============================================================================
#                          4. MAIN LOOP
# ============================================================================
def train_dino(epochs=100, batch_size=512, n_local_crops=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.25, 0.5),
        n_local_crops=n_local_crops,
        global_crop_size=96,
        local_crop_size=64
    )
    
    individual_datasets = []
    for d in DATA_DIRS:
        ds = FlatFolderDataset(root=d, transform=transform)
        if len(ds) > 0: individual_datasets.append(ds)
    combined_dataset = ConcatDataset(individual_datasets)
    print(f"Total Images (Bird Weighted): {len(combined_dataset)}")

    dataloader = DataLoader(
        combined_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=16, pin_memory=True, drop_last=True, persistent_workers=True
    )

    vit_backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0, img_size=96, dynamic_img_size=True)
    in_dim = vit_backbone.embed_dim
    model = DINO(backbone=vit_backbone, in_dim=in_dim, out_dim=65536, n_global_crops=2).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    real_model = model.module if isinstance(model, nn.DataParallel) else model
    params_groups = [
        {'params': real_model.student_backbone.parameters()}, 
        {'params': real_model.student_head.parameters()}
    ]
    
    def get_schedule(start, end, steps, warmup_steps=0):
        sched = np.concatenate((
            np.linspace(0, start, warmup_steps),
            end + 0.5 * (start - end) * (1 + np.cos(np.pi * np.arange(steps - warmup_steps) / (steps - warmup_steps)))
        ))
        return sched

    total_steps = epochs * len(dataloader)
    warmup_steps = 10 * len(dataloader)
    
    lr_max = 0.0005 * (batch_size / 256.0)
    lr_schedule = get_schedule(lr_max, 1e-6, total_steps, warmup_steps)
    
    wd_schedule = get_schedule(0.04, 0.4, total_steps)
    
    momentum_schedule = get_schedule(0.996, 1.0, total_steps)
    
    temp_schedule = get_schedule(0.04, 0.07, total_steps)

    optimizer = torch.optim.AdamW(params_groups) 
    criterion = DinoLoss(65536, temp_schedule).to(device)
    scaler = torch.amp.GradScaler('cuda')

    # Resume Logic
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        try:
            start_epoch = checkpoint['epoch'] + 1
            real_model.student_backbone.load_state_dict(checkpoint['student_backbone'])
            real_model.student_head.load_state_dict(checkpoint['student_head'])
            real_model.teacher_backbone.load_state_dict(checkpoint['teacher_backbone'])
            real_model.teacher_head.load_state_dict(checkpoint['teacher_head'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            criterion.center = checkpoint['center']
        except Exception as e:
            print(f"Resume failed (Architecture mismatch?): {e}. Starting Fresh.")
            start_epoch = 0
    else:
        print("Starting Fresh (Stage 3).")

    # Training Loop
    print("Starting training...")
    if start_epoch == 0:
        teacher_for_eval = model.module.teacher_backbone if isinstance(model, nn.DataParallel) else model.teacher_backbone
        run_knn_evaluation(teacher_for_eval, device, k=10)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        
        for i, (images, _) in enumerate(dataloader):
            step = epoch * len(dataloader) + i
            
            # Update Hyperparams per step
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[step]
                param_group["weight_decay"] = wd_schedule[step]
            
            images = [img.to(device, non_blocking=True) for img in images]
            
            with torch.amp.autocast('cuda'):
                s_out, t_out = model(images)
                loss = criterion(s_out, t_out, epoch) 

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # EMA
            m = momentum_schedule[step]
            if isinstance(model, nn.DataParallel): model.module.update_teacher_ema(m)
            else: model.update_teacher_ema(m)
            
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Ep {epoch} | Loss: {loss.item():.4f} | LR: {lr_schedule[step]:.6f}")

        print(f"--- Epoch {epoch} Done. Loss: {epoch_loss/len(dataloader):.4f} ---")
        
        # Save & Eval
        save_dict = {
            'epoch': epoch,
            'student_backbone': real_model.student_backbone.state_dict(),
            'student_head': real_model.student_head.state_dict(),
            'teacher_backbone': real_model.teacher_backbone.state_dict(),
            'teacher_head': real_model.teacher_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'center': criterion.center,
        }
        torch.save(save_dict, CHECKPOINT_PATH)
        if (epoch+1) % 5 == 0:
            torch.save(save_dict, os.path.join(CHECKPOINT_DIR, f"stage3_epoch_{epoch}.pth"))

        teacher_for_eval = model.module.teacher_backbone if isinstance(model, nn.DataParallel) else model.teacher_backbone
        run_knn_evaluation(teacher_for_eval, device, k=10)
        
        try:
             os.system(f"scp -q {CHECKPOINT_PATH} mlh9709@log-burst.hpc.nyu.edu:~/saved_checkpoints/")
        except: pass

if __name__ == "__main__":
    train_dino(epochs=100, batch_size=512, n_local_crops=8)