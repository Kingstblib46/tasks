import torch
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F
import copy
import os

# Prepare data for the attack model
def prepare_attack_data(model, iterator, device, top_k=False, test_dataset=False):
    attack_X = []
    attack_Y = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in iterator: # Assuming labels are not needed here for attack data prep based on context
            inputs = inputs.to(device)
            outputs = model(inputs)
            posteriors = F.softmax(outputs, dim=1)
            if top_k:
                topk_probs, _ = torch.topk(posteriors, 3, dim=1) # Get top 3 probabilities
                attack_X.append(topk_probs.cpu())
            else:
                attack_X.append(posteriors.cpu())

            if test_dataset: #
                attack_Y.append(torch.zeros(posteriors.size(0), dtype=torch.long)) # Label as non-member (0)
            else:
                attack_Y.append(torch.ones(posteriors.size(0), dtype=torch.long)) # Label as member (1)
    return attack_X, attack_Y

# Single epoch training
def train_per_epoch(model, train_iterator, criterion, optimizer, device, bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0
    model.train()
    for i, (features, target) in enumerate(train_iterator):
        features = features.to(device)
        target = target.to(device)
        outputs = model(features)

        if bce_loss:
            loss = criterion(outputs, target.unsqueeze(1).float()) # For BCEWithLogitsLoss, target might need to be float
        else:
            loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if not bce_loss: # Accuracy calculation for multi-class
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        else: # Accuracy calculation for binary classification (assuming outputs are logits)
            predicted = torch.sigmoid(outputs) > 0.5
            total += target.size(0)
            correct += (predicted == target.unsqueeze(1)).sum().item()


    epoch_acc = correct / total if total > 0 else 0
    epoch_loss = epoch_loss / len(train_iterator) # Average loss per batch
    return epoch_loss, epoch_acc

# Single epoch validation
def val_per_epoch(model, val_iterator, criterion, device, bce_loss=False):
    epoch_loss = 0
    epoch_acc = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (features, target) in enumerate(val_iterator):
            features = features.to(device)
            target = target.to(device)
            outputs = model(features)

            if bce_loss:
                loss = criterion(outputs, target.unsqueeze(1).float())
            else:
                loss = criterion(outputs, target)

            epoch_loss += loss.item()
            if not bce_loss: # Accuracy calculation for multi-class
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            else: # Accuracy for binary
                predicted = torch.sigmoid(outputs) > 0.5
                total += target.size(0)
                correct += (predicted == target.unsqueeze(1)).sum().item()


    epoch_acc = correct / total if total > 0 else 0
    epoch_loss = epoch_loss / len(val_iterator)
    return epoch_loss, epoch_acc

# Train attack model
def train_attack_model(model, dataset, criterion, optimizer, lr_scheduler, device,
                       model_path='./model', epochs=10, b_size=20, num_workers=1,
                       verbose=False, early_stopping=True):
    n_validation = 1000
    best_val_acc = 0
    stop_count = 0
    patience = 10 # Patience for early stopping
    path = os.path.join(model_path, 'best_attack_model.ckpt')

    train_X, train_Y = dataset
    t_X = torch.cat(train_X)
    t_Y = torch.cat(train_Y)

    attack_dataset = TensorDataset(t_X, t_Y)
    
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
        train_data, val_data = torch.utils.data.random_split(attack_dataset, [n_train_samples, n_validation_samples])
    else: # Only training data if not enough samples for validation
        train_data = attack_dataset
        val_data = None


    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=b_size, shuffle=True, num_workers=num_workers)
    
    if val_data:
        val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=b_size, shuffle=False, num_workers=num_workers)
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
                early_stopping=False, is_target=False):
    best_val_acc = 0
    patience = 5 if is_target else 10 # Different patience for target and shadow
    stop_count = 0

    target_model_filename = 'best_target_model.ckpt'
    shadow_model_filename = 'best_shadow_model.ckpt'
    
    save_path = os.path.join(model_path, target_model_filename if is_target else shadow_model_filename)

    print(f"Training {'Target' if is_target else 'Shadow'} Model...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_per_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = val_per_epoch(model, val_loader, criterion, device)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {valid_loss:.4f} Val Acc: {valid_acc:.4f}")

        if scheduler:
            scheduler.step()

        if early_stopping:
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_path)
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= patience:
                print("Early stopping triggered.")
                break
        else: # No early stopping, save if current model is better
            if best_val_acc < valid_acc:
                best_val_acc = valid_acc
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, save_path)
    
    # If not early stopping, or if early stopping didn't trigger but we want to save the last best model
    if not os.path.exists(save_path) or best_val_acc == 0: # Save last model if no model was saved yet
        if verbose: print(f"Saving model from last epoch for {'Target' if is_target else 'Shadow'} model.")
        torch.save(model.state_dict(), save_path)


    print(f"Finished training {'Target' if is_target else 'Shadow'} Model. Loading best model from: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Prepare attack data using the trained model (members from train_loader, non-members from test_loader)
    attack_X_members, attack_Y_members = prepare_attack_data(model, train_loader, device, top_k, test_dataset=False)
    
    # For non-members, use the test_loader of the *same* model (target or shadow)
    # The PDF implies using the test_loader for non-member data for the *current* model being trained.
    attack_X_non_members, attack_Y_non_members = prepare_attack_data(model, test_loader, device, top_k, test_dataset=True)
    # Note: The PDF seems to calculate test accuracy for the target/shadow model here.
    # We will return the combined data for the attack model training.

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            test_outputs = model(inputs)
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = correct / total if total > 0 else 0
        print(f"{'Target' if is_target else 'Shadow'} Model Test Accuracy: {acc*100:.2f}%")


    attack_X = attack_X_members + attack_X_non_members
    attack_Y = attack_Y_members + attack_Y_non_members

    return attack_X, attack_Y