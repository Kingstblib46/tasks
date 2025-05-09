import torch
from torch.utils.data.dataset import TensorDataset # [cite: 22]
import torch.nn.functional as F
import copy
import os

# Prepare data for the attack model
def prepare_attack_data(model, iterator, device, top_k=False, test_dataset=False): # [cite: 22]
    attack_X = [] # [cite: 23]
    attack_Y = [] # [cite: 23]
    model.eval()
    with torch.no_grad():
        for inputs, _ in iterator: # Assuming labels are not needed here for attack data prep based on context
            inputs = inputs.to(device) # [cite: 24]
            outputs = model(inputs) # [cite: 24]
            posteriors = F.softmax(outputs, dim=1) # [cite: 24]
            if top_k:
                topk_probs, _ = torch.topk(posteriors, 3, dim=1) # Get top 3 probabilities
                attack_X.append(topk_probs.cpu()) # [cite: 24]
            else:
                attack_X.append(posteriors.cpu()) # [cite: 25]

            if test_dataset: #
                attack_Y.append(torch.zeros(posteriors.size(0), dtype=torch.long)) # [cite: 25] # Label as non-member (0)
            else:
                attack_Y.append(torch.ones(posteriors.size(0), dtype=torch.long)) # [cite: 26] # Label as member (1)
    return attack_X, attack_Y

# Single epoch training
def train_per_epoch(model, train_iterator, criterion, optimizer, device, bce_loss=False): # [cite: 26]
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0
    model.train() # [cite: 27]
    for i, (features, target) in enumerate(train_iterator): # [cite: 27]
        features = features.to(device) # [cite: 27]
        target = target.to(device) # [cite: 27]
        outputs = model(features) # [cite: 27]

        if bce_loss: # [cite: 28]
            loss = criterion(outputs, target.unsqueeze(1).float()) # For BCEWithLogitsLoss, target might need to be float
        else:
            loss = criterion(outputs, target) # [cite: 28]

        optimizer.zero_grad() # [cite: 28]
        loss.backward() # [cite: 28]
        optimizer.step() # [cite: 28]

        epoch_loss += loss.item()
        if not bce_loss: # Accuracy calculation for multi-class
            _, predicted = torch.max(outputs.data, 1) # [cite: 28]
            total += target.size(0) # [cite: 29]
            correct += (predicted == target).sum().item() # [cite: 29]
        else: # Accuracy calculation for binary classification (assuming outputs are logits)
            predicted = torch.sigmoid(outputs) > 0.5
            total += target.size(0)
            correct += (predicted == target.unsqueeze(1)).sum().item()


    epoch_acc = correct / total if total > 0 else 0
    epoch_loss = epoch_loss / len(train_iterator) # Average loss per batch
    return epoch_loss, epoch_acc

# Single epoch validation
def val_per_epoch(model, val_iterator, criterion, device, bce_loss=False): # [cite: 29]
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0
    model.eval() # [cite: 30]
    with torch.no_grad(): # [cite: 30]
        for i, (features, target) in enumerate(val_iterator): # [cite: 30]
            features = features.to(device) # [cite: 30]
            target = target.to(device) # [cite: 31]
            outputs = model(features) # [cite: 31]

            if bce_loss: # [cite: 31]
                loss = criterion(outputs, target.unsqueeze(1).float()) # [cite: 31]
            else:
                loss = criterion(outputs, target) # [cite: 31]

            epoch_loss += loss.item() # [cite: 32]
            if not bce_loss: # Accuracy calculation for multi-class
                _, predicted = torch.max(outputs.data, 1) # [cite: 32]
                total += target.size(0) # [cite: 32]
                correct += (predicted == target).sum().item() # [cite: 33]
            else: # Accuracy for binary
                predicted = torch.sigmoid(outputs) > 0.5
                total += target.size(0)
                correct += (predicted == target.unsqueeze(1)).sum().item()


    epoch_acc = correct / total if total > 0 else 0 # [cite: 33]
    epoch_loss = epoch_loss / len(val_iterator) # [cite: 33]
    return epoch_loss, epoch_acc

# Train attack model
def train_attack_model(model, dataset, criterion, optimizer, lr_scheduler, device,
                       model_path='./model', epochs=10, b_size=20, num_workers=1,
                       verbose=False, early_stopping=True): # [cite: 33]
    n_validation = 1000 # [cite: 33]
    best_val_acc = 0
    stop_count = 0
    patience = 10 # Patience for early stopping
    path = os.path.join(model_path, 'best_attack_model.ckpt') # [cite: 33]

    train_X, train_Y = dataset # [cite: 33]
    t_X = torch.cat(train_X) # [cite: 34]
    t_Y = torch.cat(train_Y) # [cite: 34]

    attack_dataset = TensorDataset(t_X, t_Y) # [cite: 34]
    
    # Ensure n_validation is not larger than the dataset size
    n_total_samples = len(attack_dataset)
    if n_validation >= n_total_samples:
        print(f"Warning: n_validation ({n_validation}) is >= total samples ({n_total_samples}). Using all available samples for training and a smaller portion for validation if possible.")
        if n_total_samples > 1: # Need at least 2 samples to split
             n_validation_samples = n_total_samples // 5 # Example: use 20% for validation
             n_train_samples = n_total_samples - n_validation_samples
        else: # Not enough samples to split
            n_train_samples = n_total_samples
            n_validation_samples = 0
    else:
        n_train_samples = n_total_samples - n_validation
        n_validation_samples = n_validation


    if n_validation_samples > 0:
        train_data, val_data = torch.utils.data.random_split(attack_dataset, [n_train_samples, n_validation_samples]) # [cite: 34]
    else: # Only training data if not enough samples for validation
        train_data = attack_dataset
        val_data = None


    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=b_size, shuffle=True, num_workers=num_workers) # [cite: 35]
    
    if val_data:
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=b_size, shuffle=False, num_workers=num_workers) # [cite: 36]
    else:
        val_loader = None


    print(f"Attack Model Training - Train samples: {n_train_samples}, Val samples: {n_validation_samples if val_data else 0}")

    for i in range(epochs):
        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device, bce_loss=True) # Attack model is binary

        if val_loader:
            valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device, bce_loss=True)
            if verbose:
                print(f'Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f} Val Acc: {valid_acc:.4f}')
        else: # No validation loader
            valid_acc = train_acc # Use train_acc for best model saving if no validation
            if verbose:
                 print(f'Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')


        if lr_scheduler:
            lr_scheduler.step()

        if early_stopping and val_loader: # Early stopping only if validation is performed
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= patience:
                print("Early stopping triggered.")
                break
        elif val_loader : # No early stopping, save if current val_acc is better
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, path)
        else: # No validation, save the model from the last epoch or based on training accuracy
            if best_val_acc < train_acc: # Save based on training accuracy if no validation
                best_val_acc = train_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model,path)


    if not val_loader and best_val_acc == 0 and epochs > 0 : # If no validation, best_val_acc remains 0, save last model state
         best_model = copy.deepcopy(model.state_dict())
         torch.save(best_model, path)
         print(f"Saved model from last epoch as no validation was performed. Training accuracy: {train_acc:.4f}")
         return train_acc # Return training accuracy

    return best_val_acc


# Train target or shadow model
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
                device, model_path, verbose=False, num_epochs=50, top_k=False,
                early_stopping=False, is_target=False): # [cite: 37]
    best_val_acc = 0
    patience = 5 if is_target else 10 # Different patience for target and shadow
    stop_count = 0

    target_model_filename = 'best_target_model.ckpt'
    shadow_model_filename = 'best_shadow_model.ckpt'
    
    save_path = os.path.join(model_path, target_model_filename if is_target else shadow_model_filename) # [cite: 38]

    print(f"Training {'Target' if is_target else 'Shadow'} Model...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device) # [cite: 38]

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f} Val Acc: {valid_acc:.4f}")

        if scheduler:
            scheduler.step() # [cite: 39]

        if early_stopping:
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model_state = copy.deepcopy(model.state_dict()) # [cite: 39]
                torch.save(best_model_state, save_path) # [cite: 40, 41]
                stop_count = 0
            else:
                stop_count += 1 # [cite: 41]
            if stop_count >= patience: # [cite: 41]
                print("Early stopping triggered.")
                break
        else: # No early stopping, save if current model is better
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model_state = copy.deepcopy(model.state_dict()) # [cite: 42]
                torch.save(best_model_state, save_path) # [cite: 42]
    
    # If not early stopping, or if early stopping didn't trigger but we want to save the last best model
    if not os.path.exists(save_path) or best_val_acc == 0: # Save last model if no model was saved yet
        if verbose: print(f"Saving model from last epoch for {'Target' if is_target else 'Shadow'} model.")
        torch.save(model.state_dict(), save_path)


    print(f"Finished training {'Target' if is_target else 'Shadow'} Model. Loading best model from: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device)) # [cite: 42, 43]

    # Prepare attack data using the trained model (members from train_loader, non-members from test_loader)
    attack_X_members, attack_Y_members = prepare_attack_data(model, train_loader, device, top_k, test_dataset=False) # [cite: 43]
    
    # For non-members, use the test_loader of the *same* model (target or shadow)
    # The PDF implies using the test_loader for non-member data for the *current* model being trained.
    attack_X_non_members, attack_Y_non_members = prepare_attack_data(model, test_loader, device, top_k, test_dataset=True) # [cite: 44, 45, 46, 47]
    # Note: The PDF seems to calculate test accuracy for the target/shadow model here. [cite: 44, 45]
    # We will return the combined data for the attack model training.

    model.eval() # [cite: 44]
    with torch.no_grad(): # [cite: 44]
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device) # [cite: 44]
            labels = labels.to(device) # [cite: 44]
            test_outputs = model(inputs) # [cite: 44]
            _, predicted = torch.max(test_outputs.data, 1) # [cite: 45]
            total += labels.size(0) # [cite: 45]
            correct += (predicted == labels).sum().item() # [cite: 45]
        acc = correct / total if total > 0 else 0
        print(f"{'Target' if is_target else 'Shadow'} Model Test Accuracy: {acc*100:.2f}%")


    attack_X = attack_X_members + attack_X_non_members
    attack_Y = attack_Y_members + attack_Y_non_members

    return attack_X, attack_Y