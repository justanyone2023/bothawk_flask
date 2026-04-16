import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

def train_and_save_model(filepath, model_save_path=None):
    """
    训练模型并保存
    
    Args:
        filepath: 数据文件路径
        model_save_path: 模型保存路径，如果为None则使用默认路径
    
    Returns:
        训练好的模型和测试数据
    """
    # Load the data
    df = pd.read_csv(filepath, index_col=0)
    df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)

    # Normalize only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # Check and handle NaNs after normalization
    if df.isna().any().any():
        print("NaNs found after normalization, filling with zeros")
        df.fillna(0, inplace=True)

    df = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num','label']]
    # Splitting data
    labels = df['label']
    features = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num']]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Balance the dataset
    rus = RandomUnderSampler(random_state=42)
    train_features_rus, train_labels_rus = rus.fit_resample(train_features, train_labels)

    # Model training with hyperparameter tuning
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
    grid_search.fit(train_features_rus, train_labels_rus)

    # Save the model
    if model_save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, 'bodegha_model.pickle')
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    print(f"模型已保存到: {model_save_path}")
    
    return grid_search.best_estimator_, test_features, test_labels

def run_model_analysis(filepath):
    # Load the data
    df = pd.read_csv(filepath, index_col=0)
    df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)

    # Normalize only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


    # Check and handle NaNs after normalization
    if df.isna().any().any():
        print("NaNs found after normalization, filling with zeros")
        df.fillna(0, inplace=True)

    df = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num','label']]
    # Splitting data
    labels = df['label']
    features = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num']]
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Balance the dataset
    rus = RandomUnderSampler(random_state=42)
    train_features_rus, train_labels_rus = rus.fit_resample(train_features, train_labels)

    # Model training with hyperparameter tuning
    rfc = RandomForestClassifier()
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(rfc, param_grid=param_grid, cv=5)
    grid_search.fit(train_features_rus, train_labels_rus)

    # Evaluating the model
    y_pred = grid_search.predict(test_features)
    y_prob = grid_search.predict_proba(test_features)[:, 1]
    auc_score = roc_auc_score(test_labels, y_prob)

    accuracy = accuracy_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred, pos_label=1)
    recall = recall_score(test_labels, y_pred, pos_label=1)
    f1 = f1_score(test_labels, y_pred, pos_label=1)

    # Output evaluation results
    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score
    }

    return results

# Usage example
if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'data', 'bothawk_BoDeGHa_data.csv')
    
    # 训练并保存模型
    print("训练BoDeGHa模型...")
    model, test_features, test_labels = train_and_save_model(filepath)
    
    # 评估模型
    print("\n评估模型性能...")
    results = run_model_analysis(filepath)
    print(results)
