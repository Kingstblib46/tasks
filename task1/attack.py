import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler # [cite: 49]
# Assuming model.py and train.py are in the same directory or accessible via sys.path
import model # [cite: 49]
from model import init_params as w_init # [cite: 49]
from train import train_model, train_attack_model, prepare_attack_data # [cite: 49]
from sklearn.metrics import classification_report, precision_score, recall_score # [cite: 50]
import numpy as np
import os
import copy
import random
# import matplotlib.pyplot as plt # Not used in the core logic from PDF, can be added for visualizations

# Set seed for reproducibility
np.random.seed(1234) # [cite: 51]
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
random.seed(1234)


# --- Hyperparameters (from PDF, can be adjusted or moved to args) ---
# Model Hyperparameters
target_filters = [128, 256, 256] # [cite: 51]
shadow_filters = [64, 128, 128] # [cite: 51]
# n_fc = [256, 128] # Not explicitly used in provided TargetNet/ShadowNet, but could be for VGG classifier
num_classes_dataset = 10 # For CIFAR10/MNIST [cite: 51]
num_epochs_model = 50 # [cite: 51]
batch_size_data = 128 # [cite: 51]
learning_rate_model = 0.001 # [cite: 51]
lr_decay_model = 0.96 # [cite: 51]
reg_model = 1e-4 # Weight decay [cite: 51]
# shadow_split = 0.6 # This refers to how the overall dataset is split for target/shadow, not within get_data_loader
n_validation_samples = 1000 # [cite: 51] # Samples for validation set from test/out portion
num_workers_loader = 2 # [cite: 51]
n_hidden_mnist_model = 32 # [cite: 51]

# Attack Model Hyperparameters
NUM_EPOCHS_ATTACK = 50 # [cite: 51]
BATCH_SIZE_ATTACK = 10 # [cite: 51] # PDF uses 10, can be 128 or other values
LR_ATTACK_MODEL = 0.001 # [cite: 51]
REG_ATTACK_MODEL = 1e-7 # Weight decay for attack model [cite: 52]
LR_DECAY_ATTACK = 0.96 # [cite: 52]
n_hidden_attack = 128 # Hidden layer size for attack MLP [cite: 52]
out_classes_attack = 1 # Binary classification for attack model (member vs non-member) -- PDF specifies 2 with CrossEntropyLoss, but Sigmoid output suits 1 with BCELoss or BCEWithLogitsLoss. [cite: 52] The code uses out_classes=1 for AttackMLP with Sigmoid and train_attack_model uses bce_loss=True.


# Data transforms
def get_data_transforms(dataset_name, augm=False): # [cite: 58]
    if dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [cite: 59]
        if augm:
            train_transforms = transforms.Compose([ # [cite: 60]
                transforms.RandomRotation(5), # [cite: 60]
                transforms.RandomHorizontalFlip(p=0.5), # [cite: 60]
                transforms.RandomCrop(32, padding=4), # [cite: 60]
                transforms.ToTensor(), # [cite: 60]
                normalize]) # [cite: 60]
        else:
            train_transforms = transforms.Compose([transforms.ToTensor(), normalize]) # [cite: 60]
        test_transforms = transforms.Compose([transforms.ToTensor(), normalize]) # [cite: 59]
    elif dataset_name == 'MNIST':
        # MNIST normalization values from PyTorch examples
        normalize = transforms.Normalize((0.1307,), (0.3081,)) # [cite: 62]
        if augm:
            train_transforms = transforms.Compose([ # [cite: 63]
                transforms.RandomRotation(5), # [cite: 63]
                # transforms.RandomHorizontalFlip(p=0.5), # MNIST typically not flipped H
                transforms.ToTensor(),
                normalize])
        else:
            train_transforms = transforms.Compose([transforms.ToTensor(), normalize]) # [cite: 63]
        test_transforms = transforms.Compose([transforms.ToTensor(), normalize]) # [cite: 61, 62]
    else:
        raise ValueError("Dataset not supported")
    return train_transforms, test_transforms


# Split dataset for target and shadow model training and their respective "out" sets (non-members)
def split_dataset_indices(full_dataset_size): # [cite: 63] # Renamed to avoid conflict
    # The PDF splits the *training* set of the original dataset into four parts.
    # s_train_idx: shadow model's training members
    # s_test_idx (s_out_idx in PDF): shadow model's "out" non-members (used to generate non-member data for attack model training)
    # t_train_idx: target model's training members
    # t_test_idx (t_out_idx in PDF): target model's "out" non-members (used to generate non-member data for attack model testing)
    
    indices = list(range(full_dataset_size))
    np.random.shuffle(indices)
    
    split1 = full_dataset_size // 4
    split2 = split1 * 2
    split3 = split1 * 3
    
    s_train_idx = indices[:split1]
    s_test_idx = indices[split1:split2] # Shadow's "out" data
    t_train_idx = indices[split2:split3]
    t_test_idx = indices[split3:]       # Target's "out" data / test set for attack model's "non-member" class

    return s_train_idx, s_test_idx, t_train_idx, t_test_idx


def get_data_loaders(dataset_name, data_dir, batch_sz, augm_required=False, num_w=1, val_samples=1000): # [cite: 63]
    train_transforms, test_transforms = get_data_transforms(dataset_name, augm_required) # [cite: 64]

    if dataset_name == 'CIFAR10':
        # Load entire CIFAR10 training set first
        full_train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transforms, download=True) # [cite: 65]
        # The test set from CIFAR10 is *not* directly used for splitting according to the PDF logic for target/shadow member/non-member data.
        # It's used to generate posteriors for "non-member" data for the *attack* model.
        # However, the `split_dataset_indices` divides the `full_train_set` into four parts.
        # Let's stick to the PDF's split of the original train set.
        # `test_set` in PDF's `get_data_loader` seems to refer to the original test set of the dataset,
        # which is then used by `train_model`'s `test_loader` argument to generate non-member posteriors.
        original_test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, transform=test_transforms, download=True) # [cite: 66]

    elif dataset_name == 'MNIST':
        full_train_set = torchvision.datasets.MNIST(root=data_dir, train=True, transform=train_transforms, download=True) # [cite: 66]
        original_test_set = torchvision.datasets.MNIST(root=data_dir, train=False, transform=test_transforms, download=True) # [cite: 66]
    else:
        raise ValueError("Dataset not supported")

    s_train_idx, s_out_idx, t_train_idx, t_out_idx = split_dataset_indices(len(full_train_set)) # [cite: 66]

    # Samplers for shadow model's data
    # Shadow "in" data (members for shadow training)
    s_train_sampler = SubsetRandomSampler(s_train_idx)
    # Shadow "out" data (non-members for shadow, used for attack model training data)
    s_out_sampler = SubsetRandomSampler(s_out_idx)
    
    # Samplers for target model's data
    # Target "in" data (members for target training)
    t_train_sampler = SubsetRandomSampler(t_train_idx)
    # Target "out" data (non-members for target, used for attack model *testing* data)
    # This t_out_idx set is what the target model hasn't seen from the original train split.
    t_out_sampler = SubsetRandomSampler(t_out_idx)


    # Create validation samplers by taking a portion from the "out" sets
    # Ensure val_samples is not more than the available samples in out_idx
    
    s_val_idx = s_out_idx[:min(val_samples, len(s_out_idx))]
    t_val_idx = t_out_idx[:min(val_samples, len(t_out_idx))]
    
    s_val_sampler = SubsetRandomSampler(s_val_idx)
    t_val_sampler = SubsetRandomSampler(t_val_idx)

    # DataLoaders
    # Target model loaders
    t_train_loader = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=t_train_sampler, num_workers=num_w) # [cite: 67]
    t_val_loader = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=t_val_sampler, num_workers=num_w) # [cite: 68]
    # t_test_loader for target model (uses its "out" data portion from original train set)
    # This is the data the target model did *not* see during its training phase from the split parts.
    # This loader will be used in `train_model` (passed as `test_loader`) to get "non-member" posteriors for the *attack model's test set*.
    t_test_loader_for_attack_non_members = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=t_out_sampler, num_workers=num_w) # [cite: 68]


    # Shadow model loaders
    s_train_loader = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=s_train_sampler, num_workers=num_w) # [cite: 69]
    s_val_loader = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=s_val_sampler, num_workers=num_w) # [cite: 69]
    # s_test_loader for shadow model (uses its "out" data portion from original train set)
    # This loader will be used in `train_model` (passed as `test_loader`) to get "non-member" posteriors for the *attack model's training set*.
    s_test_loader_for_attack_non_members = torch.utils.data.DataLoader(dataset=full_train_set, batch_size=batch_sz, sampler=s_out_sampler, num_workers=num_w) # [cite: 69]

    print(f'Total Train samples in {dataset_name} (original): {len(full_train_set)}') # [cite: 69]
    print(f'Total Test samples in {dataset_name} (original): {len(original_test_set)}') # [cite: 69]
    print(f'Number of Target model train samples (members): {len(t_train_sampler)}') # [cite: 69]
    print(f'Number of Target model valid samples (from its "out" data): {len(t_val_sampler)}') # [cite: 70]
    print(f'Number of Target model test samples / "out" data (for attack non-members): {len(t_out_sampler)}') # [cite: 70]
    print(f'Number of Shadow model train samples (members): {len(s_train_sampler)}') # [cite: 70]
    print(f'Number of Shadow model valid samples (from its "out" data): {len(s_val_sampler)}') # [cite: 71]
    print(f'Number of Shadow model test samples / "out" data (for attack non-members): {len(s_out_sampler)}') # [cite: 71]


    return t_train_loader, t_val_loader, t_test_loader_for_attack_non_members, \
           s_train_loader, s_val_loader, s_test_loader_for_attack_non_members


# Attack model inference and evaluation
def attack_inference(trained_attack_model, test_X_target_posteriors, test_Y_target_labels, device, loader_num_workers=1): # [cite: 72]
    print('\n--- Attack Model Testing (on Target Model Posteriors) ---') # [cite: 72]
    target_names = ['Non-Member', 'Member'] # [cite: 72]
    
    # Convert lists of tensors to single tensors
    if isinstance(test_X_target_posteriors, list):
        X_attack_test = torch.cat(test_X_target_posteriors) # [cite: 72]
    else:
        X_attack_test = test_X_target_posteriors
        
    if isinstance(test_Y_target_labels, list):
        Y_attack_test = torch.cat(test_Y_target_labels) # [cite: 73]
    else:
        Y_attack_test = test_Y_target_labels

    attack_test_dataset = TensorDataset(X_attack_test, Y_attack_test) # [cite: 73]
    # Batch size for inference can be larger
    attack_test_loader = torch.utils.data.DataLoader(dataset=attack_test_dataset, batch_size=50, shuffle=False, num_workers=loader_num_workers) # [cite: 73]

    trained_attack_model.eval() # [cite: 73]
    
    all_pred_y = []
    all_true_y = []
    
    correct = 0
    total = 0

    with torch.no_grad(): # [cite: 73]
        for inputs, labels in attack_test_loader:
            inputs = inputs.to(device) # [cite: 73]
            labels = labels.to(device) # [cite: 73]
            
            outputs = trained_attack_model(inputs) # [cite: 73]
            
            # Assuming AttackMLP uses Sigmoid, so outputs are probabilities
            predicted_probs = outputs 
            predictions = (predicted_probs > 0.5).squeeze().long() # [cite: 74] Apply threshold

            total += labels.size(0) # [cite: 74]
            correct += (predictions == labels).sum().item() # [cite: 74]
            
            all_true_y.append(labels.cpu()) # [cite: 74]
            all_pred_y.append(predictions.cpu()) # [cite: 74]

    attack_acc = correct / total if total > 0 else 0
    print(f'Attack Test Accuracy is: {100 * attack_acc:.2f}%') # [cite: 74]

    true_y_np = torch.cat(all_true_y).numpy() # [cite: 75]
    pred_y_np = torch.cat(all_pred_y).numpy() # [cite: 75]

    print('Detailed Results for Attack Model:') # [cite: 75]
    print(classification_report(true_y_np, pred_y_np, target_names=target_names, zero_division=0)) # [cite: 75]
    precision = precision_score(true_y_np, pred_y_np, average='binary', pos_label=1, zero_division=0) # Member class is 1
    recall = recall_score(true_y_np, pred_y_np, average='binary', pos_label=1, zero_division=0) # Member class is 1
    print(f"Precision (Member): {precision:.4f}")
    print(f"Recall (Member): {recall:.4f}")


# Main function to create and run the attack
def create_attack(dataset_name, data_path_root, model_path_root,
                  train_target_model_flag, train_shadow_model_flag,
                  need_augm_flag, need_topk_posteriors, param_init_flag, verbose_flag,
                  use_early_stopping_flag=True): # [cite: 76]
    
    print(f"--- Membership Inference Attack Setup ---")
    print(f"Dataset: {dataset_name}, Data Path: {data_path_root}, Model Path: {model_path_root}")
    print(f"Train Target: {train_target_model_flag}, Train Shadow: {train_shadow_model_flag}")
    print(f"Augmentation: {need_augm_flag}, Top-K Posteriors: {need_topk_posteriors}")
    print(f"Param Init: {param_init_flag}, Verbose: {verbose_flag}, Early Stopping: {use_early_stopping_flag}")

    if dataset_name == "CIFAR10": # [cite: 76]
        img_size = 32 # [cite: 76]
        input_dim_channels = 3 # [cite: 76]
    elif dataset_name == "MNIST": # [cite: 76]
        img_size = 28 # [cite: 76] # Corrected from PDF's 26 to standard 28 for MNIST
        input_dim_channels = 1 # [cite: 76]
    else:
        raise ValueError("Dataset not supported")

    dataset_specific_data_dir = os.path.join(data_path_root, dataset_name) # [cite: 76]
    dataset_specific_model_dir = os.path.join(model_path_root, dataset_name) # [cite: 76]

    if not os.path.exists(dataset_specific_data_dir): os.makedirs(dataset_specific_data_dir, exist_ok=True) # [cite: 76]
    if not os.path.exists(dataset_specific_model_dir): os.makedirs(dataset_specific_model_dir, exist_ok=True) # [cite: 76]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # [cite: 76]
    print(f"Using device: {device}") # [cite: 76]

    # Get DataLoaders
    # Note: The shadow_split parameter in PDF's get_data_loader is handled by split_dataset_indices now
    t_train_loader, t_val_loader, t_test_loader_non_member_source, \
    s_train_loader, s_val_loader, s_test_loader_non_member_source = \
        get_data_loaders(dataset_name, dataset_specific_data_dir, batch_size_data,
                         augm_required=need_augm_flag, num_w=num_workers_loader,
                         val_samples=n_validation_samples) # [cite: 76]

    # --- Target Model ---
    print("\n--- Target Model Phase ---")
    if dataset_name == 'CIFAR10':
        target_model_instance = model.TargetNet(input_dim_channels, target_filters, img_size, num_classes_dataset).to(device) # [cite: 77]
    else: # MNIST
        target_model_instance = model.MNISTNet(input_dim_channels, n_hidden_mnist_model, num_classes_dataset, size=img_size).to(device) # [cite: 77]

    if param_init_flag: target_model_instance.apply(w_init) # [cite: 77]
    if verbose_flag: print("Target Model Architecture:\n", target_model_instance) # [cite: 77]

    target_posteriors_X = []
    target_posteriors_Y = [] # Labels for attack testing (0 for non-member, 1 for member)

    if train_target_model_flag: # [cite: 77]
        print("Training Target Model...")
        criterion_target = nn.CrossEntropyLoss() # [cite: 77]
        optimizer_target = torch.optim.Adam(target_model_instance.parameters(), lr=learning_rate_model, weight_decay=reg_model) # [cite: 77]
        scheduler_target = torch.optim.lr_scheduler.ExponentialLR(optimizer_target, gamma=lr_decay_model) # [cite: 77]
        
        # train_model returns posteriors for member (from its train_loader) and non-member (from its test_loader_non_member_source) data
        target_posteriors_X, target_posteriors_Y = train_model(
            target_model_instance, t_train_loader, t_val_loader, t_test_loader_non_member_source, # t_test_loader_non_member_source provides non-member data for attack testing
            criterion_target, optimizer_target, scheduler_target, device, dataset_specific_model_dir,
            verbose_flag, num_epochs_model, need_topk_posteriors, use_early_stopping_flag, is_target=True) # [cite: 77]
    else:
        target_file_path = os.path.join(dataset_specific_model_dir, 'best_target_model.ckpt') # [cite: 77]
        print(f"Loading pre-trained Target Model from: {target_file_path}")
        if not os.path.exists(target_file_path):
            raise FileNotFoundError(f"Target model checkpoint not found at {target_file_path}. Please train first or provide the model.")
        target_model_instance.load_state_dict(torch.load(target_file_path, map_location=device)) # [cite: 77]
        
        print("Preparing Attack Testing Data using loaded Target Model...") # [cite: 77]
        # Get member posteriors (from target's original training data part)
        t_member_X, t_member_Y = prepare_attack_data(target_model_instance, t_train_loader, device, need_topk_posteriors, test_dataset=False) # [cite: 77]
        # Get non-member posteriors (from target's "out" data part)
        t_non_member_X, t_non_member_Y = prepare_attack_data(target_model_instance, t_test_loader_non_member_source, device, need_topk_posteriors, test_dataset=True) # [cite: 77]
        target_posteriors_X = t_member_X + t_non_member_X # [cite: 77]
        target_posteriors_Y = t_member_Y + t_non_member_Y # [cite: 77]


    # --- Shadow Model ---
    print("\n--- Shadow Model Phase ---")
    if dataset_name == 'CIFAR10': # [cite: 78]
        shadow_model_instance = model.ShadowNet(input_dim_channels, shadow_filters, img_size, num_classes_dataset).to(device) # [cite: 78]
    else: # MNIST
        n_shadow_hidden_mnist = 16 # As per PDF [cite: 78]
        shadow_model_instance = model.MNISTNet(input_dim_channels, n_shadow_hidden_mnist, num_classes_dataset, size=img_size).to(device) # [cite: 78]

    if param_init_flag: shadow_model_instance.apply(w_init) # [cite: 78]
    if verbose_flag: print("Shadow Model Architecture:\n", shadow_model_instance) # [cite: 78]
    
    shadow_posteriors_for_attack_training_X = []
    shadow_posteriors_for_attack_training_Y = [] # Labels for attack training

    if train_shadow_model_flag: # [cite: 78]
        print("Training Shadow Model...")
        criterion_shadow = nn.CrossEntropyLoss() # [cite: 78]
        optimizer_shadow = torch.optim.Adam(shadow_model_instance.parameters(), lr=learning_rate_model, weight_decay=reg_model) # [cite: 78]
        scheduler_shadow = torch.optim.lr_scheduler.ExponentialLR(optimizer_shadow, gamma=lr_decay_model) # [cite: 79]
        
        shadow_posteriors_for_attack_training_X, shadow_posteriors_for_attack_training_Y = train_model(
            shadow_model_instance, s_train_loader, s_val_loader, s_test_loader_non_member_source, # s_test_loader_non_member_source provides non-member data for attack training
            criterion_shadow, optimizer_shadow, scheduler_shadow, device, dataset_specific_model_dir,
            verbose_flag, num_epochs_model, need_topk_posteriors, use_early_stopping_flag, is_target=False) # [cite: 79]
    else:
        shadow_file_path = os.path.join(dataset_specific_model_dir, 'best_shadow_model.ckpt') # [cite: 79]
        print(f"Loading pre-trained Shadow Model from: {shadow_file_path}")
        if not os.path.exists(shadow_file_path):
            raise FileNotFoundError(f"Shadow model checkpoint not found at {shadow_file_path}. Please train first or provide the model.")
        shadow_model_instance.load_state_dict(torch.load(shadow_file_path, map_location=device)) # [cite: 79]
        
        print("Preparing Attack Training Data using loaded Shadow Model...") # [cite: 79]
        # Get member posteriors (from shadow's original training data part)
        s_member_X, s_member_Y = prepare_attack_data(shadow_model_instance, s_train_loader, device, need_topk_posteriors, test_dataset=False) # [cite: 79]
        # Get non-member posteriors (from shadow's "out" data part)
        s_non_member_X, s_non_member_Y = prepare_attack_data(shadow_model_instance, s_test_loader_non_member_source, device, need_topk_posteriors, test_dataset=True) # [cite: 79]
        shadow_posteriors_for_attack_training_X = s_member_X + s_non_member_X # [cite: 79]
        shadow_posteriors_for_attack_training_Y = s_member_Y + s_non_member_Y # [cite: 79]

    # --- Attack Model ---
    print("\n--- Attack Model Phase ---")
    if not shadow_posteriors_for_attack_training_X or not shadow_posteriors_for_attack_training_Y:
        raise ValueError("Shadow model posteriors (for attack training) are empty. Ensure shadow model is trained or loaded correctly.")

    # Input size for attack model is determined by the posteriors (e.g., num_classes or top_k)
    input_size_attack = shadow_posteriors_for_attack_training_X[0].size(1) # [cite: 80]
    print(f"Input Feature dimension for Attack Model: {input_size_attack}") # [cite: 80]

    attack_model_instance = model.AttackMLP(input_size_attack, n_hidden_attack, out_classes_attack).to(device) # [cite: 80]
    if param_init_flag: attack_model_instance.apply(w_init) # [cite: 80]
    if verbose_flag: print("Attack Model Architecture:\n", attack_model_instance)
    
    # Using BCEWithLogitsLoss is generally more stable than Sigmoid + BCELoss
    # If AttackMLP has Sigmoid, use BCELoss. If it doesn't, use BCEWithLogitsLoss.
    # Our AttackMLP has Sigmoid, so BCELoss is appropriate. train_attack_model handles bce_loss=True
    criterion_attack = nn.BCELoss() if out_classes_attack == 1 else nn.CrossEntropyLoss() # [cite: 80] # The PDF uses CrossEntropyLoss for 2 classes for attack model, this code uses BCELoss for 1 output class (member prob)
    
    optimizer_attack = torch.optim.Adam(attack_model_instance.parameters(), lr=LR_ATTACK_MODEL, weight_decay=REG_ATTACK_MODEL) # [cite: 80]
    scheduler_attack = torch.optim.lr_scheduler.ExponentialLR(optimizer_attack, gamma=LR_DECAY_ATTACK) # [cite: 80]

    attack_training_dataset = (shadow_posteriors_for_attack_training_X, shadow_posteriors_for_attack_training_Y) # [cite: 80]
    
    print("Training Attack Model...")
    best_attack_val_acc = train_attack_model(
        attack_model_instance, attack_training_dataset, criterion_attack, optimizer_attack, scheduler_attack,
        device, dataset_specific_model_dir, NUM_EPOCHS_ATTACK, BATCH_SIZE_ATTACK, num_workers_loader, verbose_flag, use_early_stopping_flag) # [cite: 80]
    
    print(f"Validation Accuracy for the Best Attack Model is: {100 * best_attack_val_acc:.2f}%") # [cite: 80]

    # Load the best attack model for inference
    attack_model_path = os.path.join(dataset_specific_model_dir, 'best_attack_model.ckpt') # [cite: 80]
    if not os.path.exists(attack_model_path):
         print(f"Warning: Best attack model checkpoint not found at {attack_model_path}. Using current model state.")
    else:
        print(f"Loading best attack model from {attack_model_path}")
        attack_model_instance.load_state_dict(torch.load(attack_model_path, map_location=device)) # [cite: 80]

    # Perform inference using the trained attack model on the target model's posteriors
    if not target_posteriors_X or not target_posteriors_Y:
         raise ValueError("Target model posteriors (for attack testing) are empty. Ensure target model is trained or loaded correctly.")
    attack_inference(attack_model_instance, target_posteriors_X, target_posteriors_Y, device, loader_num_workers=num_workers_loader) # [cite: 80]

    print("\n--- Membership Inference Attack Finished ---")