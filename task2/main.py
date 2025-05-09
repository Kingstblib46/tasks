import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #This was imported but not used in PDF code
import random
from pandas.core.frame import DataFrame #This was imported but not used in PDF code

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F #This was imported but not used in PDF code

from sklearn.datasets import fetch_openml
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight #This was imported but not used in PDF code
from fairlearn.metrics import MetricFrame

# Assuming utils.py is in the same directory and contains the functions from above
import utils #This was referenced in PDF code

import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='FairML')
parser.add_argument("--epoch", default=2, type=int) # Changed to -- for consistency
parser.add_argument("--pretrain_epoch", default=1, type=int)
parser.add_argument("--method", default="base", type=str,choices=['base', 'corre', 'groupTPR', 'learn', 'remove', 'learnCorre', 'counterfactual']) # Added counterfactual from training loop
parser.add_argument("--dataset", default="adult", type=str, choices=['adult', 'pokec', 'compas', 'law'])
parser.add_argument("--s", default="sex", type=str) #sensitive attribute
parser.add_argument("--related", nargs='+', type=str, default=['race']) #related attributes, made default for testing
parser.add_argument("--r_weight", nargs='+', type=float, default=[0.1]) #related weights, made default for testing
parser.add_argument("--lr", default=0.001, type=float) # Changed from "-1r"
parser.add_argument("--weightSum", default=0.3, type=float)
parser.add_argument("--beta", default=0.5, type=float)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--model", default='MLP', type=str, choices=['MLP', 'LR', 'SVM'])

# args = parser.parse_args() # Use default args for notebook execution if not running from command line
# For testing purposes, let's create a Namespace object similar to what parse_args() would return
args = argparse.Namespace(
    epoch=2,
    pretrain_epoch=1,
    method='base', # Example method
    dataset='compas', # Using compas as per the PPT and uploaded CSV
    s='race',       # Example sensitive attribute for compas
    related=['age_cat', 'sex'], # Example related attributes for compas
    r_weight=[0.1, 0.1], # Corresponding weights
    lr=0.001,
    weightSum=0.3,
    beta=0.5,
    seed=42,
    model='MLP'
)


# 设置随机种子
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

print(f'beta: {args.beta}, weightSum: {args.weightSum}') # Used f-string

#加载数据
if args.dataset == 'adult':
    data = fetch_openml(data_id=1590, as_frame=True, parser='auto') # Added as_frame=True and parser
    data_df = data.data # Renamed to avoid conflict with module name 'data'
    if args.method == 'remove':
        for attr in args.related:
            if attr in data_df.columns: # Check if attr exists before removing
                data_df = data_df.drop(columns=[attr])
    X = pd.get_dummies(data_df)
    X = X.sort_index(axis=1)
    y_true = ((data.target == '>50K')*1).values
    n_classes = y_true.max() + 1
    data_frame = data.data.copy() # Use .copy() to avoid SettingWithCopyWarning
    sensitive_attr_data = data_frame[args.s] # Renamed to avoid conflict

    # Convert data.target to numeric if it's not already, for correlation calculation
    data_frame['target'] = (data.target == '>50K')*1


elif args.dataset == 'compas':
    # Load the uploaded Processed_Compas.csv
    try:
        data_df = pd.read_csv("Processed_Compas.csv") # [cite: 58]
    except FileNotFoundError:
        print("Error: Processed_Compas.csv not found. Please ensure it's in the same directory.")
        exit()

    # Define target and features (example, adjust as needed)
    # The PDF code doesn't explicitly show how COMPAS is preprocessed for X, y_true, sensitive_attr
    # Based on common COMPAS usage, 'is_recid' is often the target.
    # Sensitive attribute 'race' or 'sex'. Related attributes are domain specific.
    if 'is_recid' not in data_df.columns:
        print("Error: 'is_recid' column not found in Processed_Compas.csv")
        exit()
    if args.s not in data_df.columns:
        print(f"Error: Sensitive attribute '{args.s}' not found in Processed_Compas.csv")
        exit()

    y_true = data_df['is_recid'].values
    data_frame = data_df.copy() # For utils.cal_correlation and sensitive_attr
    sensitive_attr_data = data_df[args.s]

    features_to_drop = ['is_recid', args.s] # Drop target and sensitive from X
    if args.method == 'remove':
        features_to_drop.extend([attr for attr in args.related if attr in data_df.columns])

    X_raw_for_dummies = data_df.drop(columns=list(set(features_to_drop) & set(data_df.columns))) # Ensure only existing columns are dropped

    # Identify categorical and numerical columns for get_dummies (example, adjust based on your CSV)
    categorical_cols = X_raw_for_dummies.select_dtypes(include=['object', 'category']).columns
    X = pd.get_dummies(X_raw_for_dummies, columns=categorical_cols, dummy_na=False)
    X = X.sort_index(axis=1)
    n_classes = len(np.unique(y_true)) # Should be 2 for binary classification

else:
    raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")


# Common processing for adult and compas (or other future datasets)
for relate in args.related:
    if relate in data_frame.columns and args.s in data_frame.columns : #Check if columns exist
      coef = utils.cal_correlation(data_frame, args.s, relate)
      print(f'coefficient between {args.s} and {relate} is: {coef}')
    if relate in data_frame.columns and 'target' in data_frame.columns:
      # temp_df_for_target_corr = data_frame.copy() # Avoid modifying original
      # if data_frame['target'].dtype == 'object': # Ensure target is numeric for corr
      #    temp_df_for_target_corr['target_numeric'] = pd.factorize(temp_df_for_target_corr['target'])[0]
      #    target_col_for_corr = 'target_numeric'
      # else:
      #    target_col_for_corr = 'target'
      # coef_target = utils.cal_correlation(temp_df_for_target_corr, target_col_for_corr, relate)

      # Simpler approach if target is already 0/1
      coef_target = utils.cal_correlation(data_frame, 'target', relate)
      print(f'coefficient between target and {relate} is: {coef_target}')


#数据处理
indict = np.arange(X.shape[0]) # Renamed from indict to avoid clash with dict type
X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X, y_true, indict,
                                                                        test_size=0.5, stratify=y_true, random_state=7)

# Save X_train columns for counter_sample reindexing if needed by the scaler.
# processed_X_train_cols = X_train.columns.tolist() # If X_train is pandas DF before scaling

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train) # Renamed to avoid overwriting
X_test_scaled = scaler.transform(X_test)  # Renamed

# Store the original X_train (unscaled but dummified) for counter_sample if it needs original values before scaling
# This assumes data_frame for counter_sample should be the state *before* scaling & train/test split for some operations
# The PDF uses `data.data` which seems to be the raw data before dummification for `counter_sample`
# and `processed_X_train` (dummified, unscaled X_train) for `CorreErase_train`
# For `counter_sample`, it uses `data_frame` which seems to be the original dataframe (before dummify/scaling).
# Let's use `data_frame` as defined per dataset for counter_sample's X_raw argument.
# For `CorreErase_train`, it needs `processed_X_train.keys()`. Let's make `processed_X_train` the dummified X_train (unscaled)
processed_X_train_df = pd.DataFrame(X_train, columns=X.columns) # Dummified, unscaled X_train

#定义自定义 Pandas 数据集
class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes): # Python methods are usually snake_case
        # tensors = [self._df_to_tensor(df) for df in dataframes] # Corrected list comprehension
        tensors = tuple(self._df_to_tensor(df) for df in dataframes) # TensorDataset expects a tuple of tensors
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, np.ndarray):
            return torch.from_numpy(df).float()
        # return torch.from_numpy(df.values).float() # Original
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            return torch.from_numpy(df.values).float()
        return torch.tensor(df).float() # Fallback for other types like list


# 加载训练数据和测试数据
train_data = PandasDataSet(X_train_scaled, y_train, ind_train)
test_data = PandasDataSet(X_test_scaled, y_test, ind_test) #This was using X_train_scaled again in PDF

train_loader = DataLoader(train_data, batch_size=320, shuffle=True, drop_last=True)

print(f'# training samples: {len(train_data)}')
print(f'# batches: {len(train_loader)}')

#定义全连接分类器
class Classifier(nn.Module):
    def __init__(self, n_features, n_classes=2, n_hidden=32, p_dropout=0.2): # Corrected n_class to n_classes
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden * 2),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_classes), # Corrected n_class to n_classes
        )
    def forward(self, x):
        return self.network(x)

#定义线性回归分类器
class ClassifierLr(nn.Module): # Renamed from Classifier_lr
    def __init__(self, n_features, n_classes=2): # Corrected n_class
        super(ClassifierLr, self).__init__() # Corrected class name
        self.linear = nn.Linear(n_features, n_classes) # Corrected n_class
    def forward(self, x):
        return self.linear(x)

#定义SVM损失函数
def loss_svm(result, truth, model): # Renamed from loss_SVM
    truth = truth.clone() # Avoid modifying original tensor outside function if it's passed around
    truth[truth==0] = -1
    result = result.squeeze()
    weight = model.linear.weight.squeeze()
    loss = torch.mean(torch.clamp(1 - truth * result, min=0)) # truth*result for hinge loss with y in {-1,1}
    loss = loss + 0.1 * torch.mean(torch.mul(weight, weight)) # L2 regularization for SVM
    return loss

#根据不同的模型类型进行初始化
n_features = X_train_scaled.shape[1] # Use scaled shape
if args.dataset == 'pokec':
    n_hid = 72
else:
    n_hid = 32

if args.model == 'MLP':
    clf = Classifier(n_features=n_features, n_hidden=n_hid, n_classes=n_classes)
elif args.model == 'LR':
    clf = ClassifierLr(n_features=n_features, n_classes=n_classes)
elif args.model == 'SVM':
    assert n_classes == 2, "classes need to be 2 for SVM classifier"
    clf = ClassifierLr(n_features=n_features, n_classes=1) # SVM output is single value
else:
    raise NotImplementedError(f"not implemented model: {args.model}")

clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)


# 预训练分类器
def pretrain_classifier(clf, data_loader, optimizer, criterion_func): # Renamed criterion
    for x_batch, y_batch, _ in data_loader: # x, y, ind unpack
        clf.zero_grad()
        p_y = clf(x_batch)
        if args.model != 'SVM':
            loss = criterion_func(p_y, y_batch.long())
        else:
            loss = criterion_func(p_y, y_batch, clf)
        loss.backward()
        optimizer.step()
    return clf

#特征扰动训练 (Counterfactual Fairness)
def Perturb_train(clf, data_loader, optimizer, criterion_func, related_attrs_list, related_weights_list): # Renamed params
    for x_batch, y_batch, ind_batch in data_loader:
        clf.zero_grad()
        p_y = clf(x_batch)
        if args.model != 'SVM':
            loss = criterion_func(p_y, y_batch.long())
        else:
            loss = criterion_func(p_y, y_batch, clf)

        for related_attr, related_weight in zip(related_attrs_list, related_weights_list):
            # counter_sample expects X_raw (original un-dummified, unscaled), ind, related_attr, scaler
            # data_frame here refers to the original dataframe for the current dataset (e.g. compas_df)
            # ind_batch are indices relative to the *training set*, counter_sample needs indices for X_raw
            # This part needs careful handling of indices and data states.
            # The PDF uses `data.data` (likely the full original dataset before train/test split for X_raw)
            # and `ind.int()` from the loader. If ind from loader is for the *training set*,
            # it needs to be mapped back to original full dataset indices if `data.data` is the full set.
            # For simplicity, if `data_frame` is the full dataset, and `ind_train` contains original indices:
            original_indices_for_batch = ind_train[ind_batch.int().cpu().numpy()] # Map batch indices to original dataset indices

            # Ensure counter_sample is called with the correct X_raw (e.g., data_frame used in main data loading)
            # And the scaler is the one fit on X_train (dummified)
            x_new = utils.counter_sample(data_frame, original_indices_for_batch, related_attr, scaler) # X_raw is `data_frame`
            p_y_new = clf(x_new.to(x_batch.device)) # Ensure device consistency

            # p_stack = torch.stack((p_y[:,1], p_y_new[:,1]), dim=1) # Assuming binary classification and prob of class 1
            # Handle SVM case where output might be single value
            if args.model == 'SVM':
                 p_y_class1 = torch.sigmoid(p_y.squeeze()) # Convert raw scores to pseudo-probabilities if SVM
                 p_y_new_class1 = torch.sigmoid(p_y_new.squeeze())
            else:
                 p_y_class1 = p_y[:,1]
                 p_y_new_class1 = p_y_new[:,1]

            p_stack = torch.stack((p_y_class1, p_y_new_class1), dim=1)
            p_order = torch.argsort(p_stack, dim=-1)
            # cor_loss = torch.square(p_stack[:, p_order[:,1].detach()] - p_stack[:, p_order[:,0].detach()]).mean() # Original
            # Corrected indexing for p_stack based on p_order
            # This loss aims to make predictions invariant to changes in related_attr
            # It penalizes differences in predictions when related_attr is changed.
            # A simpler interpretation of the original logic: mean squared difference between p_y and p_y_new for class 1
            cor_loss = torch.square(p_y_class1 - p_y_new_class1).mean()

            loss = loss + cor_loss * related_weight
        loss.backward()
        optimizer.step()
    return clf

#相关性消除训练
def CorreErase_train(clf, data_loader, optimizer, criterion_func, related_attrs_list, related_weights_list):
    for x_batch, y_batch, ind_batch in data_loader: # ind_batch is not used here
        clf.zero_grad()
        p_y = clf(x_batch)
        if args.model != 'SVM':
            loss = criterion_func(p_y, y_batch.long())
        else:
            loss = criterion_func(p_y, y_batch, clf)

        for related_attr, related_weight in zip(related_attrs_list, related_weights_list):
            # selected_column logic from PDF is complex and assumes processed_X_train is a global or accessible df
            # It tries to find columns in x_batch that correspond to 'related_attr' after dummification.
            # Example: if related_attr='sex' and it became 'sex_Male', 'sex_Female'
            # This needs a robust way to map related_attr to its dummified column names in x_batch (which is scaled X_train_batch)
            # For now, let's assume `processed_X_train_df` (dummified, unscaled X_train) has the column names
            # And x_batch has the same feature order as X_train_scaled
            
            # Get dummified column names for the current related_attr
            # This is a simplified placeholder. A more robust way would be to pass `X.columns` or `processed_X_train_df.columns`
            # and filter them based on `related_attr`.
            # E.g. if related_attr is 'race', columns could be 'race_White', 'race_Black' etc.
            cols_for_related_attr = [col for col in processed_X_train_df.columns if related_attr in col]
            if not cols_for_related_attr:
                continue

            # Get indices of these columns in X (and thus in x_batch)
            # selected_column_indices = [X.columns.get_loc(col) for col in cols_for_related_attr if col in X.columns] # X.columns from main scope
            selected_column_indices = [processed_X_train_df.columns.get_loc(col) for col in cols_for_related_attr if col in processed_X_train_df.columns]


            if not selected_column_indices:
                continue

            # x_related_features = x_batch[:, selected_column_indices] # Features in the batch corresponding to related_attr
            # The original PDF code for cor_loss is very complex:
            # torch.sum(torch.abs(torch.mean(torch.mul(x[:,selected_column].reshape(1, x.shape[0], -1), (p_y - p_y.mean(dim=0)).transpose(0,1).reshape((-1, p_y.shape[0],1))),dim=1)))
            # This seems to be aiming for some form of covariance or correlation penalty.
            # A common way to reduce correlation is to penalize the covariance between sensitive features and predictions' residuals or predictions themselves.
            # For simplicity, let's try to penalize the magnitude of covariance between (mean of) related features and predictions.
            
            # Simplified covariance-like penalty
            # Use unscaled features for interpretability of correlation if possible, or acknowledge scaling effect
            # x_batch here is scaled. The original PDF uses x_batch directly.
            
            # For MLP/LR if p_y is logits:
            if args.model != 'SVM':
                pred_for_corr = p_y[:, 1] # Assuming prob of class 1 for binary
            else:
                pred_for_corr = torch.sigmoid(p_y.squeeze())


            for col_idx in selected_column_indices:
                # This feature in x_batch is scaled.
                # For a more direct correlation, one might use features from `processed_X_train_df` corresponding to `ind_batch`
                # However, the PDF uses x_batch directly from the loader.
                feature_col = x_batch[:, col_idx]
                
                # Center the feature and prediction
                feature_centered = feature_col - feature_col.mean()
                pred_centered = pred_for_corr - pred_for_corr.mean()
                
                covariance = (feature_centered * pred_centered).mean()
                # cor_loss_per_feature = torch.abs(covariance) # Penalize absolute covariance
                # Or square to make it differentiable and always positive
                cor_loss_per_feature = torch.square(covariance)

                loss = loss + cor_loss_per_feature * related_weight # Add penalty for each dummified feature from related_attr
        
        if torch.isnan(loss): # Check for NaN loss
            print("Warning: NaN loss encountered in CorreErase_train. Skipping backward pass for this batch.")
        else:
            loss.backward()
            optimizer.step()
    return clf


#定义群体公平性训练函数 (GroupTPR parity)
def Gfair_train(clf, data_loader, optimizer, criterion_func, related_attrs_list, related_weights_list):
    for x_batch, y_batch, ind_batch in data_loader:
        clf.zero_grad()
        p_y = clf(x_batch)
        if args.model != 'SVM':
            loss = criterion_func(p_y, y_batch.long())
        else:
            loss = criterion_func(p_y, y_batch, clf)

        for related_attr, related_weight in zip(related_attrs_list, related_weights_list):
            # groupTPR expects p_predict (probabilities), y_true, group_label, ind
            # p_y from clf might be logits. Convert to probs for groupTPR if needed.
            if args.model == 'SVM':
                p_y_probs = torch.sigmoid(p_y.detach().cpu()) # Detach and move to CPU for numpy conversion
            else: # MLP, LR
                p_y_probs = torch.softmax(p_y.detach().cpu(), dim=1)

            # group_label needs to be from the original sensitive attribute data for the current batch
            # Assuming `data_frame` holds the full dataset with original attributes
            # And `ind_train` maps training indices to original dataset indices
            original_indices_for_batch = ind_train[ind_batch.int().cpu().numpy()]
            current_batch_group_labels = data_frame[related_attr].iloc[original_indices_for_batch].values.tolist()

            # ind for groupTPR should be indices relative to the current batch (0 to batch_size-1)
            batch_internal_indices = np.arange(x_batch.shape[0])

            group_TPR_values = utils.groupTPR(p_y_probs.numpy(), y_batch.cpu().numpy(), current_batch_group_labels, batch_internal_indices)
            
            if len(group_TPR_values) >= 2: # Need at least two groups to compare
                # group_TPR_loss = torch.square(max(group_TPR_values) - min(group_TPR_values)) # Original PDF had .detach() error
                # Convert to tensor and ensure it's on the correct device
                group_TPR_tensor = torch.tensor(group_TPR_values, device=p_y.device, dtype=torch.float32)
                group_TPR_loss = torch.square(torch.max(group_TPR_tensor) - torch.min(group_TPR_tensor))
                loss = loss + group_TPR_loss * related_weight
            # else: handle cases with less than 2 groups if necessary

        if torch.isnan(loss):
             print("Warning: NaN loss encountered in Gfair_train. Skipping backward pass for this batch.")
        else:
            loss.backward()
            optimizer.step()
    return clf


# 下方图片代码需要学生手写实现: (This function is INCOMPLETE as per the PDF/PPTX)
# 相关性学习 (使用学习到的相关性权重)
def CorreLearn_train(clf, data_loader, optimizer, criterion_func, related_attrs_list, related_weights_np, weightsum_arg):
    # This function is incomplete in the provided PDF.
    # The PDF indicates "UPDATE MODEL ITERS = 1", "UPDATE WEIGHT ITERS = 1"
    # And has handwritten-like placeholders for the logic.
    # Students are expected to complete this part based on the paper's methodology.
    print("CorreLearn_train is not fully implemented as per the source PDF/PPTX.")
    print("Please complete the implementation based on the 'learnCorre' method from the reference paper.")

    # Placeholder: just run pretraining for now to make it runnable
    # clf = pretrain_classifier(clf, data_loader, optimizer, criterion_func)
    # return clf, related_weights_np # Return original weights as it's not updated
    
    # Basic structure based on the PDF's handwritten part
    UPDATE_MODEL_ITERS = 1 # From PDF image
    UPDATE_WEIGHT_ITERS = 1 # From PDF image
    
    # Convert related_weights_np to a tensor for modification if they are learned
    # This depends on whether weights are updated per batch or per epoch.
    # The PDF image implies weights are updated within the batch loop *after* model update.
    # For simplicity, let's assume weights are passed as a NumPy array and converted to tensor if needed.
    related_weights_tensor = torch.tensor(related_weights_np, dtype=torch.float32, device=next(clf.parameters()).device)


    for x_batch, y_batch, ind_batch in data_loader: # Assuming ind_batch not used here as per other functions
        # === UPDATE MODEL (similar to CorreErase_train) ===
        for _ in range(UPDATE_MODEL_ITERS):
            clf.zero_grad()
            p_y = clf(x_batch)
            if args.model != 'SVM':
                base_loss = criterion_func(p_y, y_batch.long())
            else:
                base_loss = criterion_func(p_y, y_batch, clf)
            
            total_cor_loss = 0.0
            for i, related_attr in enumerate(related_attrs_list):
                # Simplified correlation loss (example, needs to match paper's intent)
                cols_for_related_attr = [col for col in processed_X_train_df.columns if related_attr in col]
                if not cols_for_related_attr: continue
                selected_column_indices = [processed_X_train_df.columns.get_loc(col) for col in cols_for_related_attr if col in processed_X_train_df.columns]
                if not selected_column_indices: continue

                if args.model == 'SVM': pred_for_corr = torch.sigmoid(p_y.squeeze())
                else: pred_for_corr = p_y[:, 1]

                cor_loss_for_attr = 0
                for col_idx in selected_column_indices:
                    feature_col = x_batch[:, col_idx]
                    feature_centered = feature_col - feature_col.mean()
                    pred_centered = pred_for_corr - pred_for_corr.mean()
                    covariance = (feature_centered * pred_centered).mean()
                    cor_loss_for_attr += torch.square(covariance)
                
                # total_cor_loss += cor_loss_for_attr * related_weights_tensor[i] # Use the current weight
                # The image shows "loss = loss + cor_loss * related_weight * weightSum"
                # This might imply weightSum is an additional factor for the correlation part.
                total_cor_loss += cor_loss_for_attr * related_weights_tensor[i] * weightsum_arg


            loss = base_loss + total_cor_loss
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

        # === UPDATE WEIGHTS (VERY SKETCHY based on PDF image, needs proper implementation from paper) ===
        # The PDF image shows a complex weight update rule involving sorting cor_losses, beta, etc.
        # This is a critical part of the 'learnCorre' method and needs to be implemented faithfully.
        # The following is a conceptual placeholder and likely incorrect.
        with torch.no_grad(): # Weights update should not affect model gradients further
            cor_losses_items = []
            p_y_eval = clf(x_batch) # Re-evaluate predictions with updated model
            if args.model == 'SVM': pred_for_eval_corr = torch.sigmoid(p_y_eval.squeeze())
            else: pred_for_eval_corr = p_y_eval[:, 1]

            for related_attr in related_attrs_list:
                cols_for_related_attr = [col for col in processed_X_train_df.columns if related_attr in col]
                if not cols_for_related_attr:
                    cor_losses_items.append(0.0) # Placeholder if no columns found
                    continue
                selected_column_indices = [processed_X_train_df.columns.get_loc(col) for col in cols_for_related_attr if col in processed_X_train_df.columns]
                if not selected_column_indices:
                    cor_losses_items.append(0.0)
                    continue
                
                current_attr_cor_loss_sum = 0.0
                for col_idx in selected_column_indices:
                    feature_col = x_batch[:, col_idx]
                    feature_centered = feature_col - feature_col.mean()
                    pred_centered = pred_for_eval_corr - pred_for_eval_corr.mean()
                    covariance = (feature_centered * pred_centered).mean()
                    current_attr_cor_loss_sum += torch.square(covariance).item() # .item() to get Python float
                cor_losses_items.append(current_attr_cor_loss_sum)

            # The PDF's logic for weight update:
            # cor_losses_np = np.array(cor_losses_items)
            # cor_order = np.argsort(cor_losses_np) # Ascending sort of losses
            # beta_val = args.beta # beta from args
            
            # This part is highly speculative based on the poor quality image in the PDF
            # It seems to select the top k losses (smallest ones first if beta is positive, or largest if beta implies something else)
            # And distributes `weightsum_arg` among them.
            # Example: if beta = 0.5, it might be related to selecting a fraction of features.
            # The "cor_sum / (beta) / (j * (j+1))" like terms are very unclear.
            
            # --- THIS WEIGHT UPDATE MECHANISM NEEDS TO BE CORRECTLY IMPLEMENTED FROM THE REFERENCED PAPER ---
            # --- The PDF sketch is insufficient for a correct implementation. ---
            # For now, returning the weights unchanged.
            pass # Placeholder for actual weight update logic

    return clf, related_weights_tensor.cpu().numpy() # Return updated clf and current weights


#选择不同的损失函数
if args.model != 'SVM':
    clf_criterion = nn.CrossEntropyLoss()
else:
    clf_criterion = loss_svm

#预训练
for i in range(args.pretrain_epoch):
    clf.train() # Set model to training mode
    clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)

#按照不同方法训练模型
for epoch in range(args.epoch):
    clf.train() # Set model to training mode for each epoch
    if args.method == 'base':
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)
    elif args.method == 'counterfactual': # Added based on training loop structure
        clf = Perturb_train(clf, train_loader, clf_optimizer, clf_criterion, args.related, args.r_weight)
    elif args.method == 'corre':
        clf = CorreErase_train(clf, train_loader, clf_optimizer, clf_criterion, args.related, args.r_weight)
    elif args.method == 'groupTPR':
        clf = Gfair_train(clf, train_loader, clf_optimizer, clf_criterion, args.related, args.r_weight)
    elif args.method == 'learnCorre':
        # related_weights = np.array(args.r_weight) # Initialize weights
        # Ensure related_weights match the number of related attributes
        current_related_weights = np.array(args.r_weight) if len(args.r_weight) == len(args.related) else np.full(len(args.related), 1.0/len(args.related))

        clf, updated_weights = CorreLearn_train(clf, train_loader, clf_optimizer, clf_criterion,
                                        args.related, current_related_weights, args.weightSum)
        args.r_weight = updated_weights.tolist() # Update weights for next epoch if they are learned across epochs
    elif args.method == 'remove': # 'remove' method is a pre-processing step, training is like 'base'
        clf = pretrain_classifier(clf, train_loader, clf_optimizer, clf_criterion)
    # Add 'learn' method if it's different, PDF implies 'learn' might be 'learnCorre' or similar
    elif args.method == 'learn': # Assuming 'learn' is similar to 'learnCorre' or requires its own logic
        print("Warning: 'learn' method specific logic not clearly defined beyond 'learnCorre', running 'learnCorre'.")
        current_related_weights = np.array(args.r_weight) if len(args.r_weight) == len(args.related) else np.full(len(args.related), 1.0/len(args.related))
        clf, updated_weights = CorreLearn_train(clf, train_loader, clf_optimizer, clf_criterion,
                                        args.related, current_related_weights, args.weightSum)
        args.r_weight = updated_weights.tolist()


clf.eval() # Set model to evaluation mode

with torch.no_grad():
    # Ensure test_data.tensors[0] is correctly structured (it should be X_test_scaled)
    pre_clf_test = clf(torch.FloatTensor(X_test_scaled).to(next(clf.parameters()).device)) # Get raw X_test_scaled

    if args.model != 'SVM':
        y_pred_probs = torch.softmax(pre_clf_test, dim=1) # Get probabilities
        y_pred = y_pred_probs.argmax(dim=1).cpu().numpy()
    else:
        # For SVM, output is a single raw score. Threshold at 0.
        y_pred = (pre_clf_test.squeeze() > 0).int().cpu().numpy()


#计算并打印结果信息
print('sensitive attributes: ')
# Ensure sensitive_attr_data is available and correctly indexed for test set
# sensitive_attr_data was defined earlier based on the full dataset
# ind_test contains indices for the original X/y_true that went into test set
sensitive_attr_test = sensitive_attr_data.iloc[ind_test]
print(set(sensitive_attr_test))

print(f'sum of weights weightSUM for learning: {args.weightSum}')
print(f'learned lambdas: {args.r_weight}') # r_weight would be the learned lambdas for learnCorre

# Fairlearn MetricFrame
# Ensure y_test and y_pred are numpy arrays
# Ensure sensitive_attr_test is a pandas Series or 1D numpy array of same length as y_test
gm = MetricFrame(metrics=metrics.accuracy_score, # Note: metrics.accuracy_score is a single function
                   y_true=y_test,
                   y_pred=y_pred,
                   sensitive_features=sensitive_attr_test) # Use the test set's sensitive features

print(f'Average accuracy score: {gm.overall}')
print("Accuracy by group:")
print(gm.by_group)


# Calculate group selection rate and equal odds from the PDF
# This part of the PDF code for metrics seems custom.
# It calculates selection rates (positive prediction rates) per group and label,
# and "equal odds" which often refers to TPR and FPR parity.
# The PDF's "sens_eo_label" looks like TPR if label > 0.

group_selection_rate_custom = [] # Renamed to avoid conflict
group_equal_odds_custom = []

sens_test_df = sensitive_attr_test # Already a pandas Series
unique_sens_values = sens_test_df.unique()

for sens_value in unique_sens_values:
    mask_sens_value = (sens_test_df == sens_value).values # Ensure boolean mask
    
    y_sense_pred = y_pred[mask_sens_value]
    y_sense_test = y_test[mask_sens_value]
    
    sens_sr_list = [] # Renamed from sens_sr
    sens_eo_list = [] # Renamed from sens_eo
    
    for label_val in np.unique(y_test): # Iterate over unique true labels (0 and 1)
        if label_val > 0: # This condition in PDF implies TPR for label 1, what about other metrics?
            # The PDF code here calculates selection rate for y_pred==label among the current sensitive group.
            # And then a version of TPR (if label_val is positive class)
            
            # Selection Rate for current sensitive group, for current *predicted* label (if that's the intent)
            # The PDF code `(y_sense_pred==label_val).sum() / y_sense_pred.shape[0]` is selection rate of `label_val`
            # Let's assume it means selection rate of positive predictions (y_pred == 1) for the group
            if y_sense_pred.shape[0] > 0:
                 sr_positive_pred = (y_sense_pred == 1).sum() / y_sense_pred.shape[0]
            else:
                 sr_positive_pred = np.nan
            sens_sr_list.append(sr_positive_pred)

            # Equal Odds related: TPR for positive class (label_val == 1)
            # The PDF has `(y_sense_pred[y_sense_test==label_val] == label_val).sum() / (y_sense_test==label_val).sum()`
            # This is indeed TPR if label_val is the positive class (e.g. 1)
            true_positives_in_group_and_label = (y_sense_test == label_val)
            if np.sum(true_positives_in_group_and_label) > 0 :
                eo_metric = (y_sense_pred[true_positives_in_group_and_label] == label_val).sum() / np.sum(true_positives_in_group_and_label)
            else:
                eo_metric = np.nan
            sens_eo_list.append(eo_metric)
            
    if sens_sr_list: # Only append if list is not empty
        group_selection_rate_custom.append(sens_sr_list)
    if sens_eo_list: # Only append if list is not empty (i.e. label > 0 was found)
        group_equal_odds_custom.append(sens_eo_list)

group_selection_rate_np = np.array(group_selection_rate_custom)
group_equal_odds_np = np.array(group_equal_odds_custom)


if group_equal_odds_np.size > 0 :
    print('Custom Group Equal Odds (TPR for class 1 by group): ')
    print(group_equal_odds_np)
    if group_equal_odds_np.ndim > 1 and group_equal_odds_np.shape[0] > 1 and group_equal_odds_np.shape[1] > 0 : # Check shape before mean
        eo_diff = np.nanmean(np.absolute(group_equal_odds_np - np.nanmean(group_equal_odds_np, axis=0, keepdims=True)))
        print(f'eo_difference (mean absolute diff from mean TPR): {eo_diff}')
        if args.dataset == 'compas' and group_equal_odds_np.shape[0] >=3 : # Assuming at least 3 groups for this specific print
             # This assumes specific ordering/meaning of groups 0 and 2.
             # print(f'target eo difference (group 0 vs group 2): {np.absolute(group_equal_odds_np[0] - group_equal_odds_np[2])}')
             pass # This specific metric might need more context on group identity
    elif group_equal_odds_np.ndim == 1 and group_equal_odds_np.shape[0] > 1: # If it's a 1D array of TPRs per group
        eo_diff = np.nanstd(group_equal_odds_np) # Standard deviation as a measure of disparity
        print(f'eo_difference (std dev of TPRs): {eo_diff}')


if group_selection_rate_np.size > 0:
    print('Custom Group Selection Rate (Positive Prediction Rate by group): ')
    print(group_selection_rate_np)
    if group_selection_rate_np.ndim > 1 and group_selection_rate_np.shape[0] > 1 and group_selection_rate_np.shape[1] > 0:
        sr_diff = np.nanmean(np.absolute(group_selection_rate_np - np.nanmean(group_selection_rate_np, axis=0, keepdims=True)))
        print(f'sr_difference (mean absolute diff from mean SR): {sr_diff}')
        if args.dataset == 'compas' and group_selection_rate_np.shape[0] >=3:
            # print(f'target sr difference (group 0 vs group 2): {np.absolute(group_selection_rate_np[0] - group_selection_rate_np[2])}')
            pass
    elif group_selection_rate_np.ndim == 1 and group_selection_rate_np.shape[0] > 1:
        sr_diff = np.nanstd(group_selection_rate_np)
        print(f'sr_difference (std dev of SRs): {sr_diff}')


# Example of using Fairlearn's equalized_odds_difference
# This requires TPR and FPR per group.
# We have TPR from group_equal_odds_np (if it's for class 1)
# Need to calculate FPR per group: FP / (FP + TN) = FP / N_neg
# Where N_neg are true negatives.
# Or (1 - TNR)
# This part needs more careful implementation to match Fairlearn's expectations if used.