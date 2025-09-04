import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .dc_model import fit_dc, fit_dc_enhanced
from scipy.stats import poisson
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

def _probs(params: Dict[str,float], home: str, away: str, max_goals: int=10):
    # Handle missing teams by using default parameters (0.0)
    a_h=params.get(f"attack_{home}", 0.0); d_h=params.get(f"defense_{home}", 0.0)
    a_a=params.get(f"attack_{away}", 0.0); d_a=params.get(f"defense_{away}", 0.0)
    ha=params["home_adv"]; rho=params["rho"]
    lh=np.exp(a_h - d_a + ha); la=np.exp(a_a - d_h)
    ph=poisson.pmf(np.arange(0,max_goals+1),lh); pa=poisson.pmf(np.arange(0,max_goals+1),la)
    mat=np.outer(ph,pa)
    for i in (0,1):
        for j in (0,1):
            if i==0 and j==0: mat[i,j]*=(1 - lh*la*rho)
            elif i==0 and j==1: mat[i,j]*=(1 + lh*rho)
            elif i==1 and j==0: mat[i,j]*=(1 + la*rho)
            elif i==1 and j==1: mat[i,j]*=(1 - rho)
    mat/=mat.sum()
    p_home=np.tril(mat,k=-1).sum(); p_draw=np.trace(mat); p_away=np.triu(mat,k=1).sum()
    return np.array([p_home,p_draw,p_away])

def brier(y_true_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1)))

def log_loss(y_true_onehot: np.ndarray, y_prob: np.ndarray, eps: float=1e-12) -> float:
    p=np.clip(y_prob,eps,1-eps); return float(-np.mean(np.sum(y_true_onehot*np.log(p), axis=1)))

def rolling_origin(matches: pd.DataFrame, min_train: int, xi: float, max_goals: int) -> Dict[str,float]:
    """Basic rolling origin cross-validation for backward compatibility."""
    probs=[]; labs=[]
    for i in range(min_train, len(matches)):
        train=matches.iloc[:i].copy()
        test=matches.iloc[i:i+1].copy()
        params=fit_dc(train, xi=xi)
        row = test.iloc[0]
        pr = _probs(params, row["HomeTeam"], row["AwayTeam"], max_goals=max_goals)
        probs.append(pr)
        labs.append({"H":0,"D":1,"A":2}[test.iloc[0]["FTR"]])
    y_prob=np.vstack(probs); y_true=np.zeros_like(y_prob); 
    for i,lab in enumerate(labs): y_true[i,lab]=1
    return {"n":len(labs), "log_loss":log_loss(y_true,y_prob), "brier":brier(y_true,y_prob)}

def enhanced_rolling_origin(
    matches: pd.DataFrame, 
    min_train: int = 100,
    step_size: int = 10,
    xi: float = 0.001,
    max_goals: int = 10,
    use_enhanced: bool = True,
    enhanced_params: Optional[Dict] = None
) -> Dict:
    """Enhanced rolling origin cross-validation with comprehensive metrics."""
    
    if enhanced_params is None:
        enhanced_params = {
            'decay_type': 'exponential',
            'venue_effects': None,
            'use_enhanced_corr': True,
            'regularization': 0.01
        }
    
    all_probs = []
    all_labels = []
    all_dates = []
    fold_metrics = []
    
    # Ensure matches are sorted by date
    matches = matches.sort_values('Date').reset_index(drop=True)
    
    for i in range(min_train, len(matches), step_size):
        end_idx = min(i + step_size, len(matches))
        train = matches.iloc[:i].copy()
        test = matches.iloc[i:end_idx].copy()
        
        if len(test) == 0:
            continue
            
        try:
            if use_enhanced:
                params = fit_dc_enhanced(train, xi=xi, **enhanced_params)
            else:
                params = fit_dc(train, xi=xi)
                
            fold_probs = []
            fold_labels = []
            fold_dates = []
            
            for _, row in test.iterrows():
                pr = _probs(params, row["HomeTeam"], row["AwayTeam"], max_goals=max_goals)
                fold_probs.append(pr)
                fold_labels.append({"H": 0, "D": 1, "A": 2}[row["FTR"]])
                fold_dates.append(row["Date"])
            
            if fold_probs:
                fold_probs_array = np.vstack(fold_probs)
                fold_labels_array = np.array(fold_labels)
                
                # Calculate fold metrics
                fold_y_true = np.zeros((len(fold_labels), 3))
                for j, lab in enumerate(fold_labels):
                    fold_y_true[j, lab] = 1
                
                fold_accuracy = accuracy_score(fold_labels_array, np.argmax(fold_probs_array, axis=1))
                fold_brier = brier(fold_y_true, fold_probs_array)
                fold_log_loss_val = log_loss(fold_y_true, fold_probs_array)
                
                fold_metrics.append({
                    'fold': len(fold_metrics),
                    'train_size': len(train),
                    'test_size': len(test),
                    'accuracy': fold_accuracy,
                    'brier_score': fold_brier,
                    'log_loss': fold_log_loss_val,
                    'start_date': test.iloc[0]['Date'],
                    'end_date': test.iloc[-1]['Date']
                })
                
                all_probs.extend(fold_probs)
                all_labels.extend(fold_labels)
                all_dates.extend(fold_dates)
                
        except Exception as e:
            print(f"Error in fold {i}: {e}")
            continue
    
    if not all_probs:
        return {"error": "No valid predictions generated"}
    
    # Convert to arrays
    y_prob = np.vstack(all_probs)
    y_true_labels = np.array(all_labels)
    y_true_onehot = np.zeros_like(y_prob)
    for i, lab in enumerate(y_true_labels):
        y_true_onehot[i, lab] = 1
    
    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_true_labels, np.argmax(y_prob, axis=1))
    overall_brier = brier(y_true_onehot, y_prob)
    overall_log_loss = log_loss(y_true_onehot, y_prob)
    
    # Calculate precision, recall, f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, np.argmax(y_prob, axis=1), average=None, zero_division=0
    )
    
    # Calculate calibration metrics
    calibration_results = calculate_calibration_metrics(y_true_onehot, y_prob)
    
    return {
        'n_predictions': len(y_true_labels),
        'n_folds': len(fold_metrics),
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'brier_score': overall_brier,
            'log_loss': overall_log_loss,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        },
        'calibration': calibration_results,
        'fold_metrics': fold_metrics,
        'predictions': {
            'probabilities': y_prob.tolist(),
            'true_labels': y_true_labels.tolist(),
            'dates': all_dates
        }
    }

def calculate_calibration_metrics(y_true_onehot: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict:
    """Calculate calibration curve and reliability metrics."""
    calibration_data = {}
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for i, class_name in enumerate(class_names):
        y_true_class = y_true_onehot[:, i]
        y_prob_class = y_prob[:, i]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_class, y_prob_class, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob_class > bin_lower) & (y_prob_class <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true_class[in_bin].mean()
                avg_confidence_in_bin = y_prob_class[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        calibration_data[class_name.lower().replace(' ', '_')] = {
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist(),
            'expected_calibration_error': float(ece)
        }
    
    return calibration_data

def plot_calibration_curve(calibration_data: Dict, save_path: Optional[str] = None) -> None:
    """Plot calibration curves for all outcome classes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Home Win', 'Draw', 'Away Win']
    colors = ['green', 'orange', 'blue']
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        key = class_name.lower().replace(' ', '_')
        data = calibration_data[key]
        
        ax = axes[i]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(
            data['mean_predicted_value'], 
            data['fraction_of_positives'], 
            marker='o', 
            color=color,
            label=f'{class_name} (ECE: {data["expected_calibration_error"]:.3f})'
        )
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{class_name} Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def performance_over_time(fold_metrics: List[Dict], save_path: Optional[str] = None) -> None:
    """Plot model performance metrics over time."""
    df = pd.DataFrame(fold_metrics)
    df['start_date'] = pd.to_datetime(df['start_date'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy over time
    axes[0, 0].plot(df['start_date'], df['accuracy'], marker='o', color='blue')
    axes[0, 0].set_title('Accuracy Over Time')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Brier score over time
    axes[0, 1].plot(df['start_date'], df['brier_score'], marker='o', color='red')
    axes[0, 1].set_title('Brier Score Over Time')
    axes[0, 1].set_ylabel('Brier Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log loss over time
    axes[1, 0].plot(df['start_date'], df['log_loss'], marker='o', color='green')
    axes[1, 0].set_title('Log Loss Over Time')
    axes[1, 0].set_ylabel('Log Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training set size over time
    axes[1, 1].plot(df['start_date'], df['train_size'], marker='o', color='purple')
    axes[1, 1].set_title('Training Set Size Over Time')
    axes[1, 1].set_ylabel('Training Set Size')
    axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def run_comprehensive_backtest(
    matches: pd.DataFrame,
    min_train: int = 100,
    step_size: int = 10,
    xi: float = 0.001,
    max_goals: int = 10,
    enhanced_params: Optional[Dict] = None,
    save_plots: bool = False,
    plot_dir: str = './artifacts'
) -> Dict:
    """Run comprehensive backtesting with enhanced model and generate reports."""
    
    print("Running enhanced rolling origin cross-validation...")
    results = enhanced_rolling_origin(
        matches=matches,
        min_train=min_train,
        step_size=step_size,
        xi=xi,
        max_goals=max_goals,
        use_enhanced=True,
        enhanced_params=enhanced_params
    )
    
    if 'error' in results:
        return results
    
    print(f"\nBacktest Results:")
    print(f"Total predictions: {results['n_predictions']}")
    print(f"Number of folds: {results['n_folds']}")
    print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.3f}")
    print(f"Overall Brier score: {results['overall_metrics']['brier_score']:.3f}")
    print(f"Overall log loss: {results['overall_metrics']['log_loss']:.3f}")
    
    # Print class-wise metrics
    class_names = ['Home Win', 'Draw', 'Away Win']
    print("\nClass-wise metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {results['overall_metrics']['precision'][i]:.3f}")
        print(f"  Recall: {results['overall_metrics']['recall'][i]:.3f}")
        print(f"  F1-score: {results['overall_metrics']['f1_score'][i]:.3f}")
        print(f"  Support: {results['overall_metrics']['support'][i]}")
    
    # Print calibration metrics
    print("\nCalibration metrics (Expected Calibration Error):")
    for class_name in class_names:
        key = class_name.lower().replace(' ', '_')
        ece = results['calibration'][key]['expected_calibration_error']
        print(f"{class_name}: {ece:.3f}")
    
    if save_plots:
        import os
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot calibration curves
        plot_calibration_curve(
            results['calibration'], 
            save_path=f"{plot_dir}/calibration_curves.png"
        )
        
        # Plot performance over time
        performance_over_time(
            results['fold_metrics'], 
            save_path=f"{plot_dir}/performance_over_time.png"
        )
        
        print(f"\nPlots saved to {plot_dir}/")
    
    return results
