"""
性能评估脚本 - 评测各个模型的预测时间（推理时间）
"""
import time
import pandas as pd
import numpy as np
import os
import sys
import pickle
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# 添加当前目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # 项目根目录
sys.path.insert(0, SCRIPT_DIR)

# 导入各个benchmark模块
from bothawk_model_v1 import load_data as load_data_v1, preprocess_data as preprocess_data_v1
from bothawk_model_v2 import load_data as load_data_v2, preprocess_data as preprocess_data_v2, load_model as load_model_v2, predict_and_output_probabilities

# 对于rabbit，由于需要API key和网络请求，我们单独处理
# 先尝试修复sklearn兼容性问题
try:
    # 尝试导入可能缺失的sklearn内部模块
    try:
        import sklearn.ensemble._gb_losses  # type: ignore
    except (ImportError, AttributeError):
        pass
    
    try:
        import sklearn.ensemble._gb  # type: ignore
    except (ImportError, AttributeError):
        pass
    
    try:
        import sklearn.ensemble._loss  # type: ignore
    except (ImportError, AttributeError):
        pass
    
    from rabbit.rabbit import get_model as rabbit_get_model
    import joblib
    RABBIT_AVAILABLE = True
except ImportError as e:
    RABBIT_AVAILABLE = False
    print(f"Warning: Rabbit module not available: {e}")


def measure_predict_time(model, X_test, num_runs: int = 10) -> Tuple[float, float, float]:
    """
    测量模型的预测时间
    
    Args:
        model: 已训练的模型
        X_test: 测试数据
        num_runs: 运行次数，用于计算平均时间
    
    Returns:
        (平均时间, 最小时间, 最大时间)
    """
    times = []
    
    # 检查测试数据
    if len(X_test) == 0:
        print("警告: 测试数据为空")
        return 0, 0, 0
    
    # 预热（第一次预测可能较慢）
    try:
        _ = model.predict(X_test[:1])
    except Exception as e:
        print(f"预热预测失败: {e}")
        return 0, 0, 0
    
    # 使用perf_counter获得更高精度的时间测量
    # 多次测量
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            _ = model.predict(X_test)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        except Exception as e:
            print(f"预测错误: {e}")
            return 0, 0, 0
    
    if not times:
        return 0, 0, 0
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return avg_time, min_time, max_time


def measure_predict_proba_time(model, X_test, num_runs: int = 10) -> Tuple[float, float, float]:
    """
    测量模型的预测概率时间
    
    Args:
        model: 已训练的模型
        X_test: 测试数据
        num_runs: 运行次数，用于计算平均时间
    
    Returns:
        (平均时间, 最小时间, 最大时间)
    """
    times = []
    
    # 预热
    try:
        _ = model.predict_proba(X_test[:1])
    except:
        pass
    
    # 多次测量
    for _ in range(num_runs):
        start = time.time()
        try:
            _ = model.predict_proba(X_test)
            elapsed = time.time() - start
            times.append(elapsed)
        except Exception as e:
            print(f"预测概率错误: {e}")
            return 0, 0, 0
    
    if not times:
        return 0, 0, 0
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    return avg_time, min_time, max_time


def benchmark_bodegha() -> Dict:
    """评测BoDeGHa模型的预测时间"""
    print("\n" + "="*60)
    print("开始评测 BoDeGHa 模型预测时间...")
    print("="*60)
    
    model_path = os.path.join(SCRIPT_DIR, 'model', 'bodegha_model.pickle')
    data_path = os.path.join(SCRIPT_DIR, 'data', 'bothawk_BoDeGHa_data.csv')
    
    # 检查模型文件是否存在，如果不存在则训练
    if not os.path.exists(model_path):
        print("模型文件不存在，开始训练模型...")
        try:
            from BoDeGHa_bench import train_and_save_model
            model, test_features, test_labels = train_and_save_model(data_path, model_path)
        except Exception as e:
            return {
                'model': 'BoDeGHa',
                'status': 'failed',
                'error': f'训练模型失败: {str(e)}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
    
    try:
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载测试数据
        if not os.path.exists(data_path):
            return {
                'model': 'BoDeGHa',
                'status': 'failed',
                'error': f'数据文件不存在: {data_path}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
        
        import pandas as pd
        import numpy as np
        df = pd.read_csv(data_path, index_col=0)
        df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        if df.isna().any().any():
            df.fillna(0, inplace=True)
        
        features = df[['comment_type_num','comment_num','empty_comment_num','Issuecomment_num','PullRequestReviewcomment_num','Commitcomment_num']]
        from sklearn.model_selection import train_test_split
        _, X_test, _, _ = train_test_split(features, df['label'], test_size=0.3, random_state=42)
        
        # 检查测试数据
        if len(X_test) == 0:
            return {
                'model': 'BoDeGHa',
                'status': 'failed',
                'error': '测试数据为空',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0,
                'num_samples': 0
            }
        
        # 测量预测时间
        avg_time, min_time, max_time = measure_predict_time(model, X_test, num_runs=10)
        
        # 检查预测时间是否有效
        if avg_time == 0:
            try:
                test_pred = model.predict(X_test[:1])
                print(f"警告: BoDeGHa模型预测成功但测量时间为0，可能是时间过短")
            except Exception as e:
                return {
                    'model': 'BoDeGHa',
                    'status': 'failed',
                    'error': f'预测失败: {str(e)}',
                    'elapsed_time': 0,
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0,
                    'num_samples': len(X_test)
                }
        
        return {
            'model': 'BoDeGHa',
            'status': 'success',
            'elapsed_time': avg_time,
            'avg_predict_time': avg_time,
            'min_predict_time': min_time,
            'max_predict_time': max_time,
            'num_samples': len(X_test),
            'error': None
        }
    except Exception as e:
        return {
            'model': 'BoDeGHa',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }


def benchmark_bothunter() -> Dict:
    """评测Bothunter模型的预测时间"""
    print("\n" + "="*60)
    print("开始评测 Bothunter 模型预测时间...")
    print("="*60)
    
    model_path = os.path.join(SCRIPT_DIR, 'model', 'bothunter_model.pickle')
    data_path = os.path.join(SCRIPT_DIR, 'data', 'bothawk_bothunter_data.csv')
    
    # 检查模型文件是否存在，如果不存在则训练
    if not os.path.exists(model_path):
        print("模型文件不存在，开始训练模型...")
        try:
            from bothunter_bench import train_and_save_model
            model, test_features, test_labels = train_and_save_model(data_path, model_path)
        except Exception as e:
            return {
                'model': 'Bothunter',
                'status': 'failed',
                'error': f'训练模型失败: {str(e)}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
    
    try:
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载测试数据
        if not os.path.exists(data_path):
            return {
                'model': 'Bothunter',
                'status': 'failed',
                'error': f'数据文件不存在: {data_path}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
        
        import pandas as pd
        import numpy as np
        df = pd.read_csv(data_path, index_col=0)
        df['label'] = df['label'].replace({'Human': 0, 'Bot': 1}).astype(int)
        normalize = lambda x: (x - x.min()) / (x.max() - x.min())
        df = df[['name','login','bio','type','following','followers','issue_id_num','repo_id_num','unique_issue_num','unique_repo_num','unique_pr_num','issue_type_num','pr_type_num','repo_type_num','commit_type_num','label']]
        
        if df.select_dtypes(include=[np.number]).empty:
            df = df.astype(float)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        features = df[['name','login','bio','type','following','followers','issue_id_num','repo_id_num','unique_issue_num','unique_repo_num','unique_pr_num','issue_type_num','pr_type_num','repo_type_num','commit_type_num']].apply(normalize)
        from sklearn.model_selection import train_test_split
        _, X_test, _, _ = train_test_split(features, df['label'], test_size=0.3, random_state=42)
        
        # 检查测试数据
        if len(X_test) == 0:
            return {
                'model': 'Bothunter',
                'status': 'failed',
                'error': '测试数据为空',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0,
                'num_samples': 0
            }
        
        # 测量预测时间
        avg_time, min_time, max_time = measure_predict_time(model, X_test, num_runs=10)
        
        # 检查预测时间是否有效
        if avg_time == 0:
            try:
                test_pred = model.predict(X_test[:1])
                print(f"警告: Bothunter模型预测成功但测量时间为0，可能是时间过短")
            except Exception as e:
                return {
                    'model': 'Bothunter',
                    'status': 'failed',
                    'error': f'预测失败: {str(e)}',
                    'elapsed_time': 0,
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0,
                    'num_samples': len(X_test)
                }
        
        return {
            'model': 'Bothunter',
            'status': 'success',
            'elapsed_time': avg_time,
            'avg_predict_time': avg_time,
            'min_predict_time': min_time,
            'max_predict_time': max_time,
            'num_samples': len(X_test),
            'error': None
        }
    except Exception as e:
        return {
            'model': 'Bothunter',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }


def benchmark_bothawk() -> Dict:
    """评测Bothawk模型的预测时间（bothawk_model.pickle）"""
    print("\n" + "="*60)
    print("开始评测 Bothawk 模型预测时间...")
    print("="*60)
    
    model_path = os.path.join(SCRIPT_DIR, 'model', 'bothawk_model.pickle')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return {
            'model': 'Bothawk',
            'status': 'failed',
            'error': f'模型文件不存在: {model_path}',
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }
    
    try:
        # 加载模型
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # 加载测试数据
        X_train, X_test, y_train, y_test = load_data_v1()
        X_test = preprocess_data_v1(X_test)
        
        # 检查测试数据
        if len(X_test) == 0:
            return {
                'model': 'Bothawk',
                'status': 'failed',
                'error': '测试数据为空',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0,
                'num_samples': 0
            }
        
        # 测量预测时间
        avg_time, min_time, max_time = measure_predict_time(model, X_test, num_runs=10)
        
        # 检查预测时间是否有效
        if avg_time == 0:
            # 尝试单次预测以检查是否有错误
            try:
                test_pred = model.predict(X_test[:1])
                print(f"警告: Bothawk模型预测成功但测量时间为0，可能是时间过短")
            except Exception as e:
                return {
                    'model': 'Bothawk',
                    'status': 'failed',
                    'error': f'预测失败: {str(e)}',
                    'elapsed_time': 0,
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0,
                    'num_samples': len(X_test)
                }
        
        return {
            'model': 'Bothawk',
            'status': 'success',
            'elapsed_time': avg_time,  # 使用平均预测时间
            'avg_predict_time': avg_time,
            'min_predict_time': min_time,
            'max_predict_time': max_time,
            'num_samples': len(X_test),
            'error': None
        }
    except Exception as e:
        return {
            'model': 'Bothawk',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }


def benchmark_bothawk_v1() -> Dict:
    """评测Bothawk Model V1中的各个模型（包含7个bagging模型）的预测时间"""
    print("\n" + "="*60)
    print("开始评测 Bothawk Model V1（7个bagging模型）预测时间...")
    print("="*60)
    
    try:
        # 加载测试数据
        X_train, X_test, y_train, y_test = load_data_v1()
        X_test = preprocess_data_v1(X_test)
        
        # 模型名称列表
        model_names = ['DecisionTree', 'KNeighbors', 'RandomForest', 'XGBoost', 
                      'LogisticRegression', 'SVC', 'GaussianNB']
        
        model_dir = os.path.join(SCRIPT_DIR, 'model')
        sub_model_results = []
        total_time = 0
        
        # 对每个模型进行预测时间测量
        for model_name in model_names:
            model_file = f'bagging{model_name}.pickle'
            model_path = os.path.join(model_dir, model_file)
            
            if not os.path.exists(model_path):
                sub_model_results.append({
                    'model_name': model_name,
                    'status': 'failed',
                    'error': f'模型文件不存在: {model_file}',
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0
                })
                continue
            
            try:
                # 加载模型
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 测量预测时间
                avg_time, min_time, max_time = measure_predict_time(model, X_test, num_runs=10)
                total_time += avg_time
                
                sub_model_results.append({
                    'model_name': model_name,
                    'status': 'success',
                    'avg_predict_time': avg_time,
                    'min_predict_time': min_time,
                    'max_predict_time': max_time,
                    'error': None
                })
                
                print(f"  {model_name}: 平均预测时间 {avg_time:.4f}秒")
            except Exception as e:
                sub_model_results.append({
                    'model_name': model_name,
                    'status': 'failed',
                    'error': str(e),
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0
                })
        
        return {
            'model': 'Bothawk V1',
            'status': 'success',
            'elapsed_time': total_time,
            'avg_predict_time': total_time,
            'sub_models': model_names,
            'num_models': len(model_names),
            'sub_model_results': sub_model_results,
            'error': None
        }
    except Exception as e:
        return {
            'model': 'Bothawk V1',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'sub_models': []
        }


def benchmark_bothawk_v2() -> Dict:
    """评测Bothawk Model V2中的各个模型（包含3个分类器，每个有7个模型）的预测时间"""
    print("\n" + "="*60)
    print("开始评测 Bothawk Model V2（3个分类器，每个7个模型）预测时间...")
    print("="*60)
    
    try:
        # 加载测试数据
        X_train, X_test, y_train, y_test = load_data_v2()
        X_test = preprocess_data_v2(X_test)
        
        classifier_names = ['StatisticalAnalysis', 'BinaryDecision', 'Ensemble']
        model_names = ['DecisionTree', 'KNeighbors', 'RandomForest', 'XGBoost', 
                      'LogisticRegression', 'SVC', 'GaussianNB']
        
        # 准备各个分类器的测试数据
        X_statistical_test = X_test[["Number of followers", "Number of following", "tfidf_similarity", "Number of Activity",
                                     "Number of Issue", "Number of Pull Request", "Number of Repository", "Number of Commit",
                                     "Number of Active day", "Periodicity of Activities", "Number of Connection Account",
                                     "Median Response Time"]]
        
        X_binary_test = X_test[["login", "name", "email", "bio"]]
        
        model_dir = os.path.join(SCRIPT_DIR, 'model')
        classifier_results = []
        total_time = 0
        
        # 对每个分类器进行评测
        for classifier_name in classifier_names:
            print(f"\n评测分类器: {classifier_name}")
            classifier_model_results = []
            classifier_total_time = 0
            
            # 准备该分类器的测试数据
            if classifier_name == 'StatisticalAnalysis':
                X_classifier_test = X_statistical_test
            elif classifier_name == 'BinaryDecision':
                X_classifier_test = X_binary_test
            else:  # Ensemble
                # 需要先加载前两个分类器的模型来生成Ensemble的输入
                try:
                    statistical_model_path = os.path.join(model_dir, 'StatisticalAnalysis-baggingRandomForest.pickle')
                    binary_model_path = os.path.join(model_dir, 'BinaryDecision-baggingRandomForest.pickle')
                    
                    if not os.path.exists(statistical_model_path) or not os.path.exists(binary_model_path):
                        raise FileNotFoundError("需要先训练前两个分类器")
                    
                    statistical_model = load_model_v2(statistical_model_path)
                    binary_model = load_model_v2(binary_model_path)
                    
                    _, probabilities_statistical = predict_and_output_probabilities(X_statistical_test, statistical_model)
                    _, probabilities_binary = predict_and_output_probabilities(X_binary_test, binary_model)
                    
                    X_classifier_test = pd.DataFrame({
                        'StatisticalAnalysis_Prob_Positive': probabilities_statistical[:, 1],
                        'BinaryDecision_Prob_Positive': probabilities_binary[:, 1],
                    })
                except Exception as e:
                    classifier_results.append({
                        'classifier': classifier_name,
                        'status': 'failed',
                        'error': f'无法准备Ensemble测试数据: {str(e)}',
                        'elapsed_time': 0,
                        'avg_predict_time': 0,
                        'models': []
                    })
                    continue
            
            # 对每个模型进行预测时间测量
            for model_name in model_names:
                model_file = f'{classifier_name}-bagging{model_name}.pickle'
                model_path = os.path.join(model_dir, model_file)
                
                if not os.path.exists(model_path):
                    classifier_model_results.append({
                        'model_name': model_name,
                        'status': 'failed',
                        'error': f'模型文件不存在: {model_file}',
                        'avg_predict_time': 0,
                        'min_predict_time': 0,
                        'max_predict_time': 0
                    })
                    continue
                
                try:
                    # 加载模型
                    model = load_model_v2(model_path)
                    
                    # 测量预测时间
                    avg_time, min_time, max_time = measure_predict_time(model, X_classifier_test, num_runs=10)
                    classifier_total_time += avg_time
                    
                    classifier_model_results.append({
                        'model_name': model_name,
                        'status': 'success',
                        'avg_predict_time': avg_time,
                        'min_predict_time': min_time,
                        'max_predict_time': max_time,
                        'error': None
                    })
                    
                    print(f"  {model_name}: 平均预测时间 {avg_time:.4f}秒")
                except Exception as e:
                    classifier_model_results.append({
                        'model_name': model_name,
                        'status': 'failed',
                        'error': str(e),
                        'avg_predict_time': 0,
                        'min_predict_time': 0,
                        'max_predict_time': 0
                    })
            
            total_time += classifier_total_time
            classifier_results.append({
                'classifier': classifier_name,
                'status': 'success',
                'elapsed_time': classifier_total_time,
                'avg_predict_time': classifier_total_time,
                'models': classifier_model_results,
                'num_models': len([m for m in classifier_model_results if m['status'] == 'success'])
            })
        
        return {
            'model': 'Bothawk V2',
            'status': 'success',
            'elapsed_time': total_time,
            'avg_predict_time': total_time,
            'classifiers': classifier_results,
            'num_classifiers': len(classifier_names),
            'num_models_per_classifier': len(model_names),
            'error': None
        }
    except Exception as e:
        return {
            'model': 'Bothawk V2',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'classifiers': []
        }


def benchmark_rabbit() -> Dict:
    """评测Rabbit模型的预测时间，使用Rabbit的实际预测流程"""
    print("\n" + "="*60)
    print("开始评测 Rabbit 模型预测时间...")
    print("="*60)
    
    if not RABBIT_AVAILABLE:
        return {
            'model': 'Rabbit',
            'status': 'skipped',
            'error': 'Rabbit模块不可用',
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }
    
    try:
        # 使用Rabbit模块的函数加载模型和准备数据
        from rabbit.rabbit import get_model
        from rabbit.important_features import extract_features
        import pandas as pd
        import numpy as np
        
        # 加载模型，处理sklearn版本兼容性问题
        model = None
        model_path = os.path.join(SCRIPT_DIR, 'rabbit', 'bimbas.joblib')
        
        if not os.path.exists(model_path):
            return {
                'model': 'Rabbit',
                'status': 'failed',
                'error': f'模型文件不存在: {model_path}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
        
        # 方法1: 尝试使用rabbit.get_model()加载
        try:
            # 在加载前，尝试设置环境变量以处理兼容性问题
            import os as os_env
            # 尝试导入sklearn的梯度提升相关模块
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                # 尝试访问内部模块以触发导入
                _ = GradientBoostingClassifier.__module__
            except:
                pass
            
            model = get_model()
            print("使用rabbit.get_model()成功加载模型")
            # 验证模型类型
            if not hasattr(model, 'predict'):
                raise AttributeError("加载的对象不是有效的sklearn模型")
        except Exception as e1:
            error_str = str(e1)
            print(f"使用rabbit.get_model()加载失败: {error_str}")
            
            # 如果是_loss模块错误，提供更详细的解决方案
            if '_loss' in error_str:
                print("检测到_loss模块错误，这是sklearn版本兼容性问题")
                print("尝试使用兼容性加载方法...")
            # 方法2: 尝试修复_loss模块问题后加载
            try:
                import warnings
                import joblib
                import numpy as np
                
                # 尝试导入sklearn的相关模块以修复兼容性问题
                try:
                    # 尝试导入可能缺失的模块
                    import sklearn.ensemble._gb_losses  # type: ignore
                except (ImportError, AttributeError):
                    pass
                
                try:
                    import sklearn.ensemble._gb  # type: ignore
                except (ImportError, AttributeError):
                    pass
                
                try:
                    # 尝试导入_loss相关的模块
                    import sklearn.ensemble._loss  # type: ignore
                except (ImportError, AttributeError):
                    pass
                
                # 使用joblib加载，忽略警告，并尝试兼容性加载
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # 尝试使用兼容性加载（如果joblib版本支持）
                        try:
                            loaded_obj = joblib.load(model_path, mmap_mode=None)
                        except TypeError:
                            # 如果mmap_mode参数不支持，使用默认加载
                            loaded_obj = joblib.load(model_path)
                    except Exception as load_error:
                        # 如果标准加载失败，尝试使用pickle直接加载
                        print(f"joblib标准加载失败，尝试pickle: {load_error}")
                        with open(model_path, 'rb') as f:
                            loaded_obj = pickle.load(f)
                    
                    # 检查加载的对象类型
                    if hasattr(loaded_obj, 'predict'):
                        model = loaded_obj
                        print("使用joblib成功加载模型（修复兼容性后）")
                    elif isinstance(loaded_obj, dict):
                        # 可能是包含模型的字典
                        for key in ['model', 'clf', 'classifier', 'estimator']:
                            if key in loaded_obj and hasattr(loaded_obj[key], 'predict'):
                                model = loaded_obj[key]
                                print(f"从字典中提取模型成功 (key: {key})")
                                break
                    elif isinstance(loaded_obj, (list, tuple)) and len(loaded_obj) > 0:
                        # 可能是包含模型的列表或元组
                        for item in loaded_obj:
                            if hasattr(item, 'predict'):
                                model = item
                                print("从列表/元组中提取模型成功")
                                break
                    elif isinstance(loaded_obj, np.ndarray):
                        raise ValueError("加载的对象是numpy数组，不是模型对象")
                    else:
                        raise ValueError(f"无法识别加载的对象类型: {type(loaded_obj)}")
                        
            except Exception as e2:
                print(f"使用joblib加载也失败: {e2}")
                # 方法3: 尝试使用pickle加载
                try:
                    with open(model_path, 'rb') as f:
                        loaded_obj = pickle.load(f)
                    if hasattr(loaded_obj, 'predict'):
                        model = loaded_obj
                        print("使用pickle成功加载模型")
                    else:
                        raise ValueError("pickle加载的对象不是有效的模型")
                except Exception as e3:
                    # 检查是否是_loss模块错误
                    error_msgs = [str(e1), str(e2), str(e3)]
                    has_loss_error = any('_loss' in msg for msg in error_msgs)
                    
                    if has_loss_error:
                        error_msg = '模型加载失败: sklearn版本不兼容(_loss模块缺失). 解决方案: 1)重新训练模型使用当前sklearn版本, 2)降级sklearn到模型训练时的版本, 3)使用兼容的sklearn版本(建议0.24.x或1.0.x)'
                    else:
                        error_msg = f'模型加载失败: get_model失败({str(e1)[:80]}), joblib失败({str(e2)[:80]}), pickle失败({str(e3)[:80]})'
                    
                    return {
                        'model': 'Rabbit',
                        'status': 'failed',
                        'error': error_msg,
                        'elapsed_time': 0,
                        'avg_predict_time': 0,
                        'min_predict_time': 0,
                        'max_predict_time': 0
                    }
        
        if model is None or not hasattr(model, 'predict'):
            return {
                'model': 'Rabbit',
                'status': 'failed',
                'error': '模型加载失败或不是有效的sklearn模型对象',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
        
        # 尝试从已有的结果文件中获取特征数据，或创建模拟的activities数据
        # Rabbit模型需要activities DataFrame作为输入来提取特征
        try:
            # 方法1: 尝试从已有的结果文件中读取特征数据
            results_dir = os.path.join(SCRIPT_DIR, 'data', 'rabbit', 'results')
            test_features = None
            
            if os.path.exists(results_dir):
                result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
                if result_files:
                    # 尝试从第一个结果文件中读取特征
                    result_file = os.path.join(results_dir, result_files[0])
                    try:
                        result_df = pd.read_csv(result_file)
                        # 检查是否包含特征列
                        important_features = ['NA','NT','NOR','ORR',
                                              'DCA_mean','DCA_median','DCA_std','DCA_gini',
                                              'NAR_mean','NAR_median','NAR_gini','NAR_IQR',
                                              'NTR_mean','NTR_median','NTR_std','NTR_gini',
                                              'NCAR_mean','NCAR_std','NCAR_IQR',
                                              'DCAR_mean','DCAR_median','DCAR_std','DCAR_IQR',
                                              'DAAR_mean','DAAR_median','DAAR_std','DAAR_gini','DAAR_IQR',
                                              'DCAT_mean','DCAT_median','DCAT_std','DCAT_gini','DCAT_IQR',
                                              'NAT_mean','NAT_median','NAT_std','NAT_gini','NAT_IQR']
                        
                        # 检查结果文件是否包含这些特征
                        available_features = [f for f in important_features if f in result_df.columns]
                        if len(available_features) >= len(important_features) * 0.8:  # 至少80%的特征存在
                            test_features = result_df[important_features].dropna()
                            print(f"从结果文件中读取到 {len(test_features)} 条特征数据")
                    except Exception as e:
                        print(f"无法从结果文件读取特征: {e}")
            
            # 方法2: 如果没有现成的特征数据，创建模拟的activities数据并提取特征
            if test_features is None or len(test_features) == 0:
                print("未找到现成的特征数据，创建模拟activities数据...")
                # 创建模拟的activities DataFrame
                # activities需要包含: date, repository, activity等列
                n_samples = 100
                activities_data = {
                    'date': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
                    'repository': [f'repo_{i%10}' for i in range(n_samples)],
                    'activity': ['PushEvent'] * n_samples,
                    'contributor': ['test_user'] * n_samples
                }
                activities = pd.DataFrame(activities_data)
                
                # 使用Rabbit的特征提取函数
                try:
                    test_features = extract_features(activities)
                    print(f"成功提取 {len(test_features)} 条特征数据")
                except Exception as e:
                    print(f"特征提取失败: {e}")
                    # 如果特征提取失败，使用模型期望的特征数量创建数据
                    n_features = getattr(model, 'n_features_', 38)
                    important_features = ['NA','NT','NOR','ORR',
                                          'DCA_mean','DCA_median','DCA_std','DCA_gini',
                                          'NAR_mean','NAR_median','NAR_gini','NAR_IQR',
                                          'NTR_mean','NTR_median','NTR_std','NTR_gini',
                                          'NCAR_mean','NCAR_std','NCAR_IQR',
                                          'DCAR_mean','DCAR_median','DCAR_std','DCAR_IQR',
                                          'DAAR_mean','DAAR_median','DAAR_std','DAAR_gini','DAAR_IQR',
                                          'DCAT_mean','DCAT_median','DCAT_std','DCAT_gini','DCAT_IQR',
                                          'NAT_mean','NAT_median','NAT_std','NAT_gini','NAT_IQR']
                    test_features = pd.DataFrame(
                        np.random.rand(n_samples, min(n_features, len(important_features))),
                        columns=important_features[:min(n_features, len(important_features))]
                    )
            
            # 检查测试数据
            if test_features is None or len(test_features) == 0:
                return {
                    'model': 'Rabbit',
                    'status': 'failed',
                    'error': '无法准备测试数据',
                    'elapsed_time': 0,
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0,
                    'num_samples': 0
                }
            
            # 确保特征顺序正确（按照模型期望的顺序）
            n_features = getattr(model, 'n_features_', None)
            if n_features and len(test_features.columns) != n_features:
                # 如果特征数量不匹配，尝试调整
                if len(test_features.columns) > n_features:
                    test_features = test_features.iloc[:, :n_features]
                else:
                    # 如果特征不足，用0填充
                    missing = n_features - len(test_features.columns)
                    for i in range(missing):
                        test_features[f'feature_{len(test_features.columns)}'] = 0
            
            # 尝试预测
            try:
                # 先测试单次预测
                test_pred = model.predict_proba(test_features[:1])
                print(f"Rabbit模型测试预测成功，预测结果形状: {test_pred.shape if hasattr(test_pred, 'shape') else 'N/A'}")
                
                # 测量预测时间（使用predict_proba，因为Rabbit实际使用predict_proba）
                times = []
                n_runs = 10
                
                # 预热
                try:
                    _ = model.predict_proba(test_features[:1])
                except:
                    pass
                
                # 多次测量
                for _ in range(n_runs):
                    start = time.perf_counter()
                    try:
                        _ = model.predict_proba(test_features)
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                    except Exception as e:
                        print(f"预测错误: {e}")
                        return {
                            'model': 'Rabbit',
                            'status': 'failed',
                            'error': f'预测失败: {str(e)}',
                            'elapsed_time': 0,
                            'avg_predict_time': 0,
                            'min_predict_time': 0,
                            'max_predict_time': 0,
                            'num_samples': len(test_features)
                        }
                
                if not times:
                    return {
                        'model': 'Rabbit',
                        'status': 'failed',
                        'error': '无法测量预测时间',
                        'elapsed_time': 0,
                        'avg_predict_time': 0,
                        'min_predict_time': 0,
                        'max_predict_time': 0,
                        'num_samples': len(test_features)
                    }
                
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                print(f"Rabbit模型预测时间测量成功: 平均={avg_time:.6f}秒, 最小={min_time:.6f}秒, 最大={max_time:.6f}秒")
                
                return {
                    'model': 'Rabbit',
                    'status': 'success',
                    'elapsed_time': avg_time,
                    'avg_predict_time': avg_time,
                    'min_predict_time': min_time,
                    'max_predict_time': max_time,
                    'num_samples': len(test_features),
                    'error': None
                }
            except Exception as pred_error:
                error_msg = str(pred_error)
                print(f"Rabbit模型预测失败: {error_msg}")
                return {
                    'model': 'Rabbit',
                    'status': 'failed',
                    'error': f'预测失败: {error_msg}',
                    'elapsed_time': 0,
                    'avg_predict_time': 0,
                    'min_predict_time': 0,
                    'max_predict_time': 0,
                    'num_samples': len(test_features) if 'test_features' in locals() else 0
                }
        except Exception as e:
            return {
                'model': 'Rabbit',
                'status': 'failed',
                'error': f'准备测试数据失败: {str(e)}',
                'elapsed_time': 0,
                'avg_predict_time': 0,
                'min_predict_time': 0,
                'max_predict_time': 0
            }
    except Exception as e:
        return {
            'model': 'Rabbit',
            'status': 'failed',
            'error': str(e),
            'elapsed_time': 0,
            'avg_predict_time': 0,
            'min_predict_time': 0,
            'max_predict_time': 0
        }


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 0.01:
        # 对于非常小的时间，使用毫秒或微秒显示
        if seconds < 0.001:
            return f"{seconds * 1000000:.2f}微秒"
        else:
            return f"{seconds * 1000:.2f}毫秒"
    elif seconds < 60:
        return f"{seconds:.4f}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}分{secs:.2f}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}小时{minutes}分{secs:.2f}秒"


def print_results(results: List[Dict]):
    """打印性能评估结果（预测时间）"""
    print("\n" + "="*80)
    print("模型预测时间评估结果汇总")
    print("="*80)
    
    # 创建结果表格
    table_data = []
    for result in results:
        avg_time = result.get('avg_predict_time', result.get('elapsed_time', 0))
        min_time = result.get('min_predict_time', 0)
        max_time = result.get('max_predict_time', 0)
        
        row = {
            '模型': result['model'],
            '状态': result['status'],
            '平均预测时间': format_time(avg_time),
            '平均预测时间(秒)': f"{avg_time:.4f}" if avg_time > 0 else 'N/A',
            '最小预测时间(秒)': f"{min_time:.4f}" if min_time > 0 else 'N/A',
            '最大预测时间(秒)': f"{max_time:.4f}" if max_time > 0 else 'N/A'
        }
        
        # Bothawk V1的特殊处理
        if result['model'] == 'Bothawk V1' and result.get('num_models'):
            row['子模型数量'] = result['num_models']
            if avg_time > 0:
                row['平均时间/模型'] = format_time(avg_time / result['num_models'])
        
        # Bothawk V2的特殊处理
        if result['model'] == 'Bothawk V2' and result.get('classifiers'):
            row['分类器数量'] = result['num_classifiers']
            row['每个分类器模型数'] = result['num_models_per_classifier']
            total_models = result['num_classifiers'] * result['num_models_per_classifier']
            row['总模型数'] = total_models
            if avg_time > 0:
                row['平均时间/模型'] = format_time(avg_time / total_models)
        
        if result.get('num_samples'):
            row['测试样本数'] = result['num_samples']
        
        if result.get('error'):
            row['错误信息'] = result['error'][:50] + '...' if len(result['error']) > 50 else result['error']
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # 打印Bothawk V1的详细子模型信息
    for result in results:
        if result['model'] == 'Bothawk V1' and result.get('sub_model_results'):
            print("\n" + "="*80)
            print("Bothawk V1 各子模型详细预测时间")
            print("="*80)
            sub_model_data = []
            for sub_model in result['sub_model_results']:
                sub_model_data.append({
                    '模型名称': sub_model['model_name'],
                    '状态': sub_model['status'],
                    '平均预测时间': format_time(sub_model['avg_predict_time']),
                    '平均预测时间(秒)': f"{sub_model['avg_predict_time']:.4f}",
                    '最小预测时间(秒)': f"{sub_model['min_predict_time']:.4f}",
                    '最大预测时间(秒)': f"{sub_model['max_predict_time']:.4f}"
                })
            sub_model_df = pd.DataFrame(sub_model_data)
            print(sub_model_df.to_string(index=False))
    
    # 打印Bothawk V2的详细分类器信息
    for result in results:
        if result['model'] == 'Bothawk V2' and result.get('classifiers'):
            print("\n" + "="*80)
            print("Bothawk V2 各分类器详细预测时间")
            print("="*80)
            classifier_data = []
            for classifier in result['classifiers']:
                classifier_data.append({
                    '分类器': classifier['classifier'],
                    '状态': classifier['status'],
                    '总预测时间': format_time(classifier.get('avg_predict_time', classifier.get('elapsed_time', 0))),
                    '总预测时间(秒)': f"{classifier.get('avg_predict_time', classifier.get('elapsed_time', 0)):.4f}",
                    '成功模型数': classifier.get('num_models', 0)
                })
            classifier_df = pd.DataFrame(classifier_data)
            print(classifier_df.to_string(index=False))
            
            # 打印每个分类器下的模型详情
            for classifier in result['classifiers']:
                if classifier.get('models'):
                    print(f"\n{classifier['classifier']} 分类器下各模型预测时间:")
                    model_data = []
                    for model in classifier['models']:
                        model_data.append({
                            '模型名称': model['model_name'],
                            '状态': model['status'],
                            '平均预测时间(秒)': f"{model['avg_predict_time']:.4f}",
                            '最小预测时间(秒)': f"{model['min_predict_time']:.4f}",
                            '最大预测时间(秒)': f"{model['max_predict_time']:.4f}"
                        })
                    model_df = pd.DataFrame(model_data)
                    print(model_df.to_string(index=False))
    
    # 保存详细结果到CSV
    output_file = os.path.join(SCRIPT_DIR, 'result', 'performance_benchmark_results.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 准备保存的数据
    save_data = []
    for result in results:
        avg_time = result.get('avg_predict_time', result.get('elapsed_time', 0))
        min_time = result.get('min_predict_time', 0)
        max_time = result.get('max_predict_time', 0)
        
        save_row = {
            'model': result['model'],
            'status': result['status'],
            'avg_predict_time_seconds': avg_time,
            'min_predict_time_seconds': min_time,
            'max_predict_time_seconds': max_time,
            'avg_predict_time_formatted': format_time(avg_time),
            'num_samples': result.get('num_samples', 'N/A')
        }
        
        if result.get('error'):
            save_row['error'] = result['error']
        
        # Bothawk V1的特殊处理
        if result['model'] == 'Bothawk V1' and result.get('num_models'):
            save_row['num_sub_models'] = result['num_models']
            if avg_time > 0:
                save_row['avg_time_per_model'] = avg_time / result['num_models']
            save_row['sub_models'] = ', '.join(result.get('sub_models', []))
            # 保存每个子模型的详细时间
            if result.get('sub_model_results'):
                for i, sub_model in enumerate(result['sub_model_results']):
                    save_row[f'sub_model_{i+1}_name'] = sub_model['model_name']
                    save_row[f'sub_model_{i+1}_avg_time'] = sub_model['avg_predict_time']
                    save_row[f'sub_model_{i+1}_min_time'] = sub_model['min_predict_time']
                    save_row[f'sub_model_{i+1}_max_time'] = sub_model['max_predict_time']
                    save_row[f'sub_model_{i+1}_status'] = sub_model['status']
        
        # Bothawk V2的特殊处理
        if result['model'] == 'Bothawk V2' and result.get('classifiers'):
            save_row['num_classifiers'] = result['num_classifiers']
            save_row['num_models_per_classifier'] = result['num_models_per_classifier']
            total_models = result['num_classifiers'] * result['num_models_per_classifier']
            save_row['total_models'] = total_models
            if avg_time > 0:
                save_row['avg_time_per_model'] = avg_time / total_models
            # 保存每个分类器的时间
            for i, classifier in enumerate(result['classifiers']):
                save_row[f'classifier_{i+1}_name'] = classifier['classifier']
                save_row[f'classifier_{i+1}_avg_time'] = classifier.get('avg_predict_time', classifier.get('elapsed_time', 0))
                save_row[f'classifier_{i+1}_status'] = classifier['status']
                save_row[f'classifier_{i+1}_num_models'] = classifier.get('num_models', 0)
        
        save_data.append(save_row)
    
    save_df = pd.DataFrame(save_data)
    save_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 性能排名（按预测时间，从快到慢）
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] in ['failed', 'skipped']]
    
    if successful_results:
        print("\n" + "="*80)
        print("预测时间排名（从快到慢）")
        print("="*80)
        sorted_results = sorted(successful_results, key=lambda x: x.get('avg_predict_time', x.get('elapsed_time', float('inf'))))
        for i, result in enumerate(sorted_results, 1):
            avg_time = result.get('avg_predict_time', result.get('elapsed_time', 0))
            min_time = result.get('min_predict_time', 0)
            max_time = result.get('max_predict_time', 0)
            # 显示更详细的时间信息
            if avg_time > 0:
                time_str = format_time(avg_time)
                if min_time > 0 and max_time > 0:
                    print(f"{i}. {result['model']}: {time_str} (平均: {avg_time:.6f}秒, 最小: {min_time:.6f}秒, 最大: {max_time:.6f}秒)")
                else:
                    print(f"{i}. {result['model']}: {time_str} (平均: {avg_time:.6f}秒)")
            else:
                print(f"{i}. {result['model']}: 0.00秒 (可能预测失败或时间过短)")
    
    # 显示失败或跳过的模型
    if failed_results:
        print("\n" + "="*80)
        print("未成功评测的模型")
        print("="*80)
        for result in failed_results:
            status_str = "跳过" if result['status'] == 'skipped' else "失败"
            error_msg = result.get('error', '未知错误')
            print(f"- {result['model']}: {status_str} - {error_msg}")


def main():
    """主函数"""
    print("="*80)
    print("模型预测时间评估工具")
    print("="*80)
    print("开始评测各个模型的预测时间（推理时间）...")
    print("注意: 本工具评估的是模型的预测时间，而非训练时间")
    print("="*80)
    
    results = []
    
    # 评测各个模型
    results.append(benchmark_bodegha())
    results.append(benchmark_bothunter())
    results.append(benchmark_bothawk())
    results.append(benchmark_bothawk_v1())
    results.append(benchmark_bothawk_v2())
    results.append(benchmark_rabbit())
    
    # 打印结果
    print_results(results)
    
    print("\n" + "="*80)
    print("预测时间评估完成！")
    print("="*80)


if __name__ == '__main__':
    main()


