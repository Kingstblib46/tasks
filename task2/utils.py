import torch
import numpy as np
import copy
import pandas as pd
import random
from sklearn.cluster import KMeans
import numbers

#计算各组的TPR (True Positive Rate)
def groupTPR(p_predict, y_true, group_label, ind):
    group_set = set(group_label)
    # if len(group_set) > 5 and isinstance (list (group_set) [0], numbers.Number): # Original code had an error here
    if len(group_set) > 5 and isinstance (list(group_set)[0], numbers.Number):
        # kmeans = KMeans (n_clusters=5, random_state=0).fit (group_label.reshape(-1,1)) # Original code had an error here
        kmeans = KMeans (n_clusters=5, random_state=0, n_init='auto').fit(np.array(group_label).reshape(-1,1))
        # group_label = kmeans.predict (group_label.reshape(-1,1)) # Original code had an error here
        group_label = kmeans.predict(np.array(group_label).reshape(-1,1))
        group_set = set(group_label)
        group_label = group_label[ind.int()]
    else:
        group_label = np.array(group_label)[ind.int()] # Ensure group_label is an array for consistent indexing
        group_set = set(group_label)

    group_tpr = []
    for group_val in group_set: # Renamed 'group' to 'group_val' to avoid conflict
        group_true_ind = np.array([a==1 and b==group_val for a,b in zip(y_true,group_label)])
        if np.sum(group_true_ind) > 0: # Ensure there are true positives in the group
            cur_tpr = p_predict[group_true_ind,:][:,1].mean()
            if not np.isnan(cur_tpr): # Changed from cur_tpr.isnan() to np.isnan(cur_tpr)
                group_tpr.append(cur_tpr)
        # else: # Optional: handle cases with no true positives in a group
        #     group_tpr.append(np.nan) # Or 0, or skip
    return group_tpr

# 计算各组的TNR (True Negative Rate) # Changed from FNR in comment based on function name groupTNR
def groupTNR (p_predict, y_true, group_label, ind): # The PDF calls this groupTNR but implements groupFNR logic
    group_set = set(group_label)
    # if len(group_set) > 5 and isinstance (list (group_set) [0], numbers.Number): # Original code had an error here
    if len(group_set) > 5 and isinstance (list(group_set)[0], numbers.Number):
        # kmeans = KMeans (n_clusters=5, random_state=0).fit(group_label.reshape(-1,1)) # Original code had an error here
        kmeans = KMeans (n_clusters=5, random_state=0, n_init='auto').fit(np.array(group_label).reshape(-1,1))
        # group_label = kmeans.predict(group_label.reshape(-1,1)) # Original code had an error here
        group_label = kmeans.predict(np.array(group_label).reshape(-1,1))
        group_set = set(group_label)
        group_label = group_label[ind.int()]
    else:
        group_label = np.array(group_label)[ind.int()] # Ensure group_label is an array for consistent indexing
        group_set = set(group_label)

    group_tnr = [] # Changed from group_fnr based on function name
    for group_val in group_set: # Renamed 'group' to 'group_val'
        # group_true_ind = np.array([a==0 and b==group_val for a,b in zip(y_true,group_label)]) # Original logic for FNR (a==0 for TN)
        group_false_ind = np.array([a==0 and b==group_val for a,b in zip(y_true,group_label)]) # TNR: True is 0
        if np.sum(group_false_ind) > 0: # Ensure there are true negatives in the group
            # cur_fnr = p_predict[group_true_ind,:][:,0].mean() # Original logic for FNR
            cur_tnr = p_predict[group_false_ind,:][:,0].mean() # TNR: Predicted is 0 (index 0)
            if not np.isnan(cur_tnr): # Changed from cur_fnr.isnan() to np.isnan(cur_tnr)
                group_tnr.append(cur_tnr)
        # else: # Optional
        #     group_tnr.append(np.nan)
    return group_tnr


#生成对抗样本,通过修改特定属性生成新的数据
def counter_sample(X_raw, ind, related_attr, scaler):
    X_new = copy.deepcopy(X_raw)
    attr_candid = list(set(X_raw[related_attr]))
    attr_new = random.choices(attr_candid, k=ind.shape[0])
    X_new.loc[ind, related_attr] = attr_new
    X_new = pd.get_dummies(X_new)
    # Ensure columns are sorted to match the scaler's expectation if not already
    # This is important if get_dummies changes column order/set from original X_raw used for scaler fitting
    # If scaler was fit on X after get_dummies and sort_index, this might not be an issue,
    # but it's safer to ensure column order consistency.
    # Assuming X_raw passed to this function is the one *before* get_dummies
    # And the scaler was fit on data that *was* dummified and sorted.
    # We need to ensure X_new has all columns the scaler expects.
    # This part can be tricky if the dummified columns change due to new 'attr_new' values.
    # A robust way would be to reindex X_new with the columns the scaler was trained on.
    # For now, sticking to the original code structure:
    X_new = X_new.sort_index(axis=1)
    # The following line is problematic: X_new.iloc[ind] might not have the same columns as X_train if new dummy vars were created
    # It should be X_new transformed entirely then select rows by ind, or ensure X_new has same columns as scaler expects
    # Assuming scaler expects the full dummified and sorted feature set.
    X_new_transformed_all = scaler.transform(X_new) # Transform the whole dataframe
    X_new_selected_rows = X_new_transformed_all[ind] # Then select the relevant rows
    return torch.FloatTensor(X_new_selected_rows) # Original code returns X_new.iloc[ind] transformed.


# 计算敏感属性和相关属性之间的相关性
def cal_correlation(X_raw, sens_attr, related_attr): #renamed from cal correlation
    X_src = pd.get_dummies(X_raw[[sens_attr]]) # Pass as list to ensure it's a DataFrame
    X_relate = pd.get_dummies(X_raw[[related_attr]]) # Pass as list
    correffics = []
    for i in range(len(X_src.keys())):
        for j in range(len(X_relate.keys())):
            # correffic = abs(X_src[X_src.keys()[i]].corr(X_relate[X_relate.keys()[j]])) # Original
            # Handle cases where a column might be constant after get_dummies (e.g., if an attribute has only one value in selection)
            # .corr() can return NaN if one of the series is constant.
            val_src = X_src[X_src.keys()[i]]
            val_relate = X_relate[X_relate.keys()[j]]
            if val_src.nunique() < 2 or val_relate.nunique() < 2:
                correffic = 0 # Or handle as NaN, but 0 might be safer for sum
            else:
                correffic = abs(val_src.corr(val_relate))

            if not np.isnan(correffic): # Ensure NaN is not added
                 correffics.append(correffic)
            else:
                 correffics.append(0) # Add 0 if NaN to prevent sum issues

    return sum(correffics)