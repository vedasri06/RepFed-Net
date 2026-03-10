import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, f1_score
import matplotlib.pyplot as plt
import tenseal as ts

torch._dynamo.config.suppress_errors = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RetinalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise OSError(f"Image file does not exist: {img_path}")
        for mask_path in mask_paths:
            if not os.path.exists(mask_path):
                raise OSError(f"Mask file does not exist: {mask_path}")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
mask_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset directories
client_directories = [
    {"image_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client1\train\image"], "mask_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client1\train\mask"]},
    {"image_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client2\train\image"], "mask_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client2\train\mask"]},
    {"image_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client3\train\image"], "mask_dirs": [r"D:\RETINAL_SEGMENTATION\retinalsegmentation\client3\train\mask"]}
]

client_datasets, val_datasets, test_datasets = [], [], []
for client_dir in client_directories:
    image_paths, mask_paths = [], []
    for img_dir in client_dir["image_dirs"]:
        image_paths.extend([os.path.join(img_dir, fname) for fname in sorted(os.listdir(img_dir))])
    for mask_dir in client_dir["mask_dirs"]:
        mask_paths.extend([os.path.join(mask_dir, fname) for fname in sorted(os.listdir(mask_dir))])
    if len(image_paths) != len(mask_paths):
        raise ValueError("Mismatch in image and mask counts")
    full_dataset = RetinalDataset(image_paths, mask_paths, image_transform=data_transform, mask_transform=mask_transform)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    client_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    test_datasets.append(test_dataset)

def print_dataset_sizes(datasets, dataset_name):
    for i, dataset in enumerate(datasets):
        print(f"{dataset_name} Client {i+1}: {len(dataset)} samples")

print_dataset_sizes(client_datasets, "Train")
print_dataset_sizes(val_datasets, "Validation")
print_dataset_sizes(test_datasets, "Test")

batch_size = 32
train_loaders = [DataLoader(d, batch_size=batch_size, shuffle=True) for d in client_datasets]
val_loaders = [DataLoader(d, batch_size=batch_size, shuffle=False) for d in val_datasets]
test_loaders = [DataLoader(d, batch_size=batch_size, shuffle=False) for d in test_datasets]

# Model architecture 
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, filters, activation='relu'):
        super(InceptionBlock, self).__init__()
        filter1x1, filter3x3, filter5x5, reduce3x3, reduce5x5, pool_proj = filters
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1x1, kernel_size=1, padding=0),
            nn.BatchNorm2d(filter1x1),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce3x3, kernel_size=1, padding=0),
            nn.BatchNorm2d(reduce3x3),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity(),
            nn.Conv2d(reduce3x3, filter3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter3x3),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce5x5, kernel_size=1, padding=0),
            nn.BatchNorm2d(reduce5x5),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity(),
            nn.Conv2d(reduce5x5, filter5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(filter5x5),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        )
        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, padding=0),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.Identity()
        )
    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        pool_proj = self.pool_proj(x)
        return torch.cat([conv1x1, conv3x3, conv5x5, pool_proj], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False) if in_channels != out_channels else nn.Identity()
        self.se = SEBlock(out_channels)
    def forward(self, x):
        out = self.conv(x)
        residual = self.residual(x)
        out = out + residual
        out = self.se(out)
        return out

class PyramidAttention(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(PyramidAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(F_g, F_int, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(F_int, F_g, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        self.combine = nn.Sequential(
            nn.Conv2d(F_int * 3 + F_g, F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_l),
            nn.Sigmoid()
        )
    def forward(self, g, x):
        b, c, _, _ = g.size()
        global_context = self.global_pool(g).view(b, c)
        global_weight = self.global_fc(global_context).view(b, c, 1, 1)
        global_out = g * global_weight
        c1 = self.conv1x1(x)
        c3 = self.conv3x3(x)
        c5 = self.conv5x5(x)
        combined = torch.cat([global_out, c1, c3, c5], dim=1)
        attention = self.combine(combined)
        return x * attention

class UNetInceptionAttentionRes(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetInceptionAttentionRes, self).__init__()
        f_1, f_2, f_3, f_4 = 32, 64, 128, 256
        self.o1 = InceptionBlock(in_channels, [f_1//4]*6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.o2 = InceptionBlock(sum([f_1//4]*4), [f_2//4]*6)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.o3 = InceptionBlock(sum([f_2//4]*4), [f_3//4]*6)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.o4 = InceptionBlock(sum([f_3//4]*4), [f_4//4]*6)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = InceptionBlock(sum([f_4//4]*4), [512//4]*6)
        self.up4 = nn.ConvTranspose2d(sum([512//4]*4), sum([f_4//4]*4), kernel_size=2, stride=2)
        self.att4 = PyramidAttention(F_g=sum([f_4//4]*4), F_l=sum([f_4//4]*4), F_int=sum([f_4//4]*2))
        self.dec4 = ConvBlock(sum([f_4//4]*8), sum([f_4//4]*4))
        self.up3 = nn.ConvTranspose2d(sum([f_4//4]*4), sum([f_3//4]*4), kernel_size=2, stride=2)
        self.att3 = PyramidAttention(F_g=sum([f_3//4]*4), F_l=sum([f_3//4]*4), F_int=sum([f_3//4]*2))
        self.dec3 = ConvBlock(sum([f_3//4]*8), sum([f_3//4]*4))
        self.up2 = nn.ConvTranspose2d(sum([f_3//4]*4), sum([f_2//4]*4), kernel_size=2, stride=2)
        self.att2 = PyramidAttention(F_g=sum([f_2//4]*4), F_l=sum([f_2//4]*4), F_int=sum([f_2//4]*2))
        self.dec2 = ConvBlock(sum([f_2//4]*8), sum([f_2//4]*4))
        self.up1 = nn.ConvTranspose2d(sum([f_2//4]*4), sum([f_1//4]*4), kernel_size=2, stride=2)
        self.att1 = PyramidAttention(F_g=sum([f_1//4]*4), F_l=sum([f_1//4]*4), F_int=sum([f_1//4]*2))
        self.dec1 = ConvBlock(sum([f_1//4]*8), sum([f_1//4]*4))
        self.out = nn.Conv2d(sum([f_1//4]*4), out_channels, kernel_size=1)
    
    def forward(self, x):
        s1 = self.o1(x)
        p1 = self.pool1(s1)
        s2 = self.o2(p1)
        p2 = self.pool2(s2)
        s3 = self.o3(p2)
        p3 = self.pool3(s3)
        s4 = self.o4(p3)
        p4 = self.pool4(s4)
        b = self.bottleneck(p4)
        d4 = self.up4(b)
        s4 = self.att4(g=d4, x=s4)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        s3 = self.att3(g=d3, x=s3)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        s2 = self.att2(g=d2, x=s2)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        s1 = self.att1(g=d1, x=s1)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)
        output = self.out(d1)
       
        output = torch.clamp(output, min=-10, max=10)
        return torch.sigmoid(output)

    def calculate_accuracy(self, pred, target):
        pred = (pred > 0.5).float()
        correct = (pred == target).float()
        return correct.sum().item() / correct.numel()

    def calculate_iou(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = (pred + target).sum(dim=(2, 3)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.sum().item()

def evaluate(model, loader):
    model.eval()
    total_loss, total_iou, total_acc, total = 0, 0, 0, 0
    all_labels, all_predictions, all_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
           
            outputs = torch.clamp(outputs, 0, 1)
            loss = nn.BCELoss()(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            total_iou += model.calculate_iou(preds, labels)
            total_acc += model.calculate_accuracy(preds, labels) * inputs.size(0)
            total += inputs.size(0)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())
    return (total_loss/total, total_iou/total, total_acc/total,
            all_labels, all_predictions, all_probs)

def calculate_f1_score(labels, predictions, probs):

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    return f1

def show_predictions(model, dataset, index=0):
    model.eval()
    image, mask = dataset[index]
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_mask = model(image)[0].cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8)
    image = image.cpu().numpy()[0].transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

class Client:
    def __init__(self, train_dataset, val_loader, test_loader, client_id, batch_size, ckks_context):
        self.dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_loader.dataset
        self.client_id = client_id
        self.model = UNetInceptionAttentionRes(in_channels=3, out_channels=1).to(device)
        self.ckks_context = ckks_context
        self.current_f1 = 0.0 
        self.previous_smoothed_f1 = 0.0

    def train(self, criterion, optimizer):
        self.model.train()
        running_loss, running_iou, running_acc, total = 0, 0, 0, 0
        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
        
            outputs = torch.clamp(outputs, 0, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_iou += self.model.calculate_iou(preds, labels)
            running_acc += self.model.calculate_accuracy(preds, labels) * inputs.size(0)
            total += inputs.size(0)
        train_loss = running_loss / total
        train_iou = running_iou / total
        train_acc = running_acc / total
        val_loss, val_iou, val_acc, val_labels, val_preds, val_probs = evaluate(self.model, self.val_loader)
       
   
        val_f1 = calculate_f1_score(val_labels, val_preds, val_probs)
        self.current_f1 = val_f1 
       
        print(f'{self.client_id}: Train Loss = {train_loss:.4f}, Train IoU = {train_iou:.4f}, Train Acc = {train_acc:.4f}, '
              f'Val Loss = {val_loss:.4f}, Val IoU = {val_iou:.4f}, Val Acc = {val_acc:.4f}, Val F1 = {val_f1:.4f}')
        return train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, val_f1

    def get_encrypted_weights_and_f1(self):
       
        weights = self.model.state_dict()
        encrypted_weights = {}
        encrypted_f1 = None
        enc_time = 0.0
       
  
        start = time.time()
        try:
            encrypted_f1 = ts.ckks_vector(self.ckks_context, [self.current_f1])
        except:
            encrypted_f1 = None
        enc_time += time.time() - start
       
   
        for name, param in weights.items():
            param_flat = param.cpu().numpy().flatten()
            if 'out.' in name:
                start = time.time()
                try:
                    encrypted_param = ts.ckks_vector(self.ckks_context, param_flat.tolist())
                    enc_time += time.time() - start
                    encrypted_weights[name] = (encrypted_param, param.shape)
                except:
                    encrypted_weights[name] = (param_flat.tolist(), param.shape)
            else:
                encrypted_weights[name] = (param_flat.tolist(), param.shape)
               
        return encrypted_weights, encrypted_f1, enc_time

    def set_model_weights(self, weights):

        state_dict = self.model.state_dict()
        dec_time = 0.0
        for name, param in weights.items():
            if isinstance(param, torch.Tensor):
                state_dict[name].copy_(param.to(device))
            else:
      
                if isinstance(param, dict):
                    param_data, shape = param['data'], param['shape']
                    if isinstance(param_data, list):
                        param_tensor = torch.tensor(param_data).reshape(shape).to(device)
                    else:
                        param_tensor = torch.tensor(param_data).to(device)
                    state_dict[name].copy_(param_tensor)
                else:
                    state_dict[name].copy_(param.to(device))
        self.model.load_state_dict(state_dict)
        return dec_time

    def test(self):
        test_loss, test_iou, test_acc, labels, predictions, probs = evaluate(self.model, self.test_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        auc = roc_auc_score(labels, probs)
        conf_matrix = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f'{self.client_id}: Test Loss = {test_loss:.4f}, Test IoU = {test_iou:.4f}, Test Acc = {test_acc:.4f}')
        print(f'{self.client_id}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}, '
              f'Specificity = {specificity:.4f}, AUC = {auc:.4f}')
        print(f'{self.client_id}: Confusion Matrix:\n{conf_matrix}')
        return test_loss, test_iou, test_acc, precision, recall, f1, specificity, auc, labels, predictions, probs

    def show_predictions(self, num_samples=3):
        print(f"\nPredictions for {self.client_id}:")
        for i in range(min(num_samples, len(self.test_dataset))):
            show_predictions(self.model, self.test_dataset, index=i)


context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()


num_clients = 3
rounds = 100  
learning_rate = 0.001
num_epochs_per_round = 20  
smoothing_factor = 0.7

clients = [Client(client_datasets[i], val_loaders[i], test_loaders[i], f'Client {i+1}', batch_size, context) for i in range(num_clients)]


smoothed_reputations = [0.3] * num_clients  
history = []

print("=== Starting  Training with Reputation-Aware Aggregation ===")

for round_num in range(rounds):
    round_start_time = time.time()
    print(f'\n=== Starting Round {round_num + 1}/{rounds} ===')
   
    client_losses, client_ious, client_accs = [], [], []
    val_losses, val_ious, val_accs = [], [], []
    client_f1_scores = []


    for i, client in enumerate(clients):
        optimizer = torch.optim.Adam(client.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        for epoch in range(num_epochs_per_round):
            train_loss, train_iou, train_acc, val_loss, val_iou, val_acc, val_f1 = client.train(nn.BCELoss(), optimizer)
            client_losses.append(train_loss)
            client_ious.append(train_iou)
            client_accs.append(train_acc)
            val_losses.append(val_loss)
            val_ious.append(val_iou)
            val_accs.append(val_acc)
            client_f1_scores.append(val_f1)


    current_f1_scores = [client.current_f1 for client in clients]
    total_reputation = sum(smoothed_reputations)
    
    for i in range(num_clients):
        smoothed_reputations[i] = (smoothing_factor * smoothed_reputations[i] +
                                  (1 - smoothing_factor) * current_f1_scores[i])
    
 
    total_rep = sum(smoothed_reputations)
    if total_rep == 0:
        reputation_weights = [1.0 / num_clients] * num_clients
    else:
        reputation_weights = [r / total_rep for r in smoothed_reputations]
   
    print(f"Round {round_num + 1} Reputation Weights: {[f'{w:.4f}' for w in reputation_weights]}")
    print(f"Round {round_num + 1} Smoothed Reputation Scores: {[f'{r:.4f}' for r in smoothed_reputations]}")

   
    print('\n--- Reputation-Aware Aggregation Phase ---')
    global_state = None
    
    for i, client in enumerate(clients):
        client_weights = client.model.state_dict()
        rep_weight = reputation_weights[i]
        
        if global_state is None:
            global_state = {name: param.clone() * rep_weight for name, param in client_weights.items()}
        else:
            for name in global_state.keys():
                global_state[name] += client_weights[name] * rep_weight

 
    total_weight = sum(reputation_weights)
    for name in global_state:
        global_state[name] /= total_weight

   
    total_dec_time = 0.0
    for client in clients:
        dec_start = time.time()
        client.model.load_state_dict(global_state)
        total_dec_time += time.time() - dec_start


    print('\n--- Evaluating Global Model on Test Sets ---')
    all_labels, all_predictions, all_probs = [], [], []
    global_test_results = []
    
    for client in clients:
        result = client.test()
        (_, _, _, _, _, _, _, _, labels, predictions, probs) = result
        all_labels.extend(labels)
        all_predictions.extend(predictions)
        all_probs.extend(probs)
        global_test_results.append(result)


    global_precision, global_recall, global_f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0)
    global_auc = roc_auc_score(all_labels, all_probs)
    global_conf = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = global_conf.ravel()
    global_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n=== Global Test Metrics (Reputation-Aware) ===")
    print(f"Precision = {global_precision:.4f}, Recall = {global_recall:.4f}, F1-Score = {global_f1:.4f}, "
          f"Specificity = {global_specificity:.4f}, AUC = {global_auc:.4f}")
    print(f"Global Confusion Matrix:\n{global_conf}")


    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Global ROC (AUC = {global_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Global ROC Curve - Round {round_num + 1} (Reputation-Aware)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


    round_total_time = time.time() - round_start_time
    print(f'\nRound {round_num + 1}: Total Model Update Time = {total_dec_time:.4f} seconds')
    print(f'Round {round_num + 1}: Total Round Time = {round_total_time:.4f} seconds')

    history.append({
        "round": round_num + 1,
        "avg_train_loss": np.mean(client_losses[-num_clients:]),
        "avg_train_iou": np.mean(client_ious[-num_clients:]),
        "avg_train_acc": np.mean(client_accs[-num_clients:]),
        "avg_val_loss": np.mean(val_losses[-num_clients:]),
        "avg_val_iou": np.mean(val_ious[-num_clients:]),
        "avg_val_acc": np.mean(val_accs[-num_clients:]),
        "global_precision": global_precision,
        "global_recall": global_recall,
        "global_f1": global_f1,
        "global_specificity": global_specificity,
        "global_auc": global_auc,
        "reputation_weights": reputation_weights.copy(),
        "round_time": round_total_time
    })

# Save final model
os.makedirs(r"D:\RETINAL_SEGMENTATION\DRIVE_results", exist_ok=True)
torch.save(clients[0].model.state_dict(), r"D:\RETINAL_SEGMENTATION\DRIVE_results\repfed_net_reputation_aware_fixed.pth")
print(r"\n RepFed-Net model saved to D:\RETINAL_SEGMENTATION\DRIVE_results\repfed_net_reputation_aware_fixed.pth")

# Show final predictions
print("\n=== Final Predictions ===")
for client in clients:
    client.show_predictions(num_samples=3)









