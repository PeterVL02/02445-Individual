import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLUMNS = ['Round', 'Phase', 'Individual', 'Puzzler', 'Frustrated', 'Cohort']
NON_CATEGORICAL_COLUMNS = ['HR_Mean', 'HR_Median', 'HR_std', 'HR_Min', 'HR_Max', 'HR_AUC']


def stratified_k_fold_split(X, y, k, random_state=None):
    """
    Split the dataset into k stratified folds.
    
    Parameters:
    X (pd.DataFrame or np.array): Feature set.
    y (pd.Series or np.array): Target variable.
    k (int): Number of folds.
    
    Returns:
    list of tuples: Each tuple contains train and test indices for one fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds



def individual_folds(X: pd.DataFrame, y: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split the dataset into folds where each fold consists of all observations referring to one individual.
    
    Parameters:
    X (pd.DataFrame): Feature set.
    y (pd.Series or np.array): Target variable.
    
    Returns:
    list of tuples: Each tuple contains train and test indices for one fold.
    """
    # Identify the individual columns
    individual_columns = [col for col in X.columns if col.startswith('Individual')]
    
    # Create a DataFrame to store individual information
    individual_info_list = []
    for col in individual_columns:
        individual_id = int(col.split('_')[1])
        individual_info_list.append(pd.DataFrame({
            'index': X.index[X[col] == 1],
            'individual': individual_id
        }))
    
    individual_info = pd.concat(individual_info_list)
    
    unique_individuals = individual_info['individual'].unique()
    
    folds = []
    
    # Create folds for each individual
    for individual in unique_individuals:
        train_inds = individual_info[individual_info['individual'] != individual]['index']
        test_inds = individual_info[individual_info['individual'] == individual]['index']
        
        X_train, X_test = X.loc[train_inds], X.loc[test_inds]
        y_train, y_test = y[train_inds], y[test_inds]
        
        folds.append((X_train, X_test, y_train, y_test))
    
    return folds

def custom_k_fold_split(x, y, method, k=None, random_state=None):
    if method == 'stratified':
        return stratified_k_fold_split(x, y, k, random_state)
    elif method == 'individual':
        return individual_folds(x, y)
    
def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize the specified non-categorical columns in the training and test datasets.
    
    Parameters:
    X_train (pd.DataFrame): Training feature set.
    X_test (pd.DataFrame): Test feature set.
    
    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: Standardized training and test datasets.
    """
    global NON_CATEGORICAL_COLUMNS
    scaler = StandardScaler()
    
    for col in NON_CATEGORICAL_COLUMNS:
        if col in X_train.columns and col in X_test.columns:
            X_train[col] = scaler.fit_transform(X_train[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))
    
    return X_train, X_test

def load_data(file_path: str):
    data = pd.read_csv(file_path, index_col=0).reset_index(drop=True)
    categorical_columns = ['Round', 'Phase', 'Individual', 'Puzzler', 'Frustrated', 'Cohort']

    data_encoded = pd.get_dummies(data, columns=categorical_columns)

    target_columns = [col for col in data_encoded.columns if 'Frustrated_' in col]

    y = data_encoded[target_columns]

    X = data_encoded.drop(columns=target_columns)


    y = np.argmax(y.values, axis=1)
    return X, y