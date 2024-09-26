import time 
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def prepare_experiment(log_path):
    """Prepares folder under log_path directory to save experiments results.

    Args:
        log_path: (String) Directory of log folder.
    Returns:
        Tuple of experiment ID and log directory.
    """
    exp_id = int(time.time())
    log_dir = os.path.join(log_path, '{}'.format(exp_id))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    model_log_dir = os.path.join(log_dir, 'model')
    if not os.path.exists(model_log_dir):
        os.makedirs(model_log_dir) 
    return exp_id, log_dir

def aggregate_results(results):
    """Helper function to aggregate a list of dictionaries into a dictionary of lists.
    Keys are used to collect the results from multiple dictionaries into a single list.

    Args:
        results (list): List of dictionaries.

    Returns:
        Dictionary: Dictionary of lists.
    """
    result = {}
    for key in results[0].keys():
        result[key] = [r[key] for r in results]
    return result
        
def print_result(result, name='vanilla'):
    """Prints the cols keys in the result dictionary. name is prepended
       to cols before selection.

    Args:
        result: (Dictionary) Metrics are labels, values are list of results.
        name: (String) Prepended every string in cols.
    Returns:
        None
    """
    cols = ['prevalence', 'coverage', 'sensitivity', 'specificity', 'ppv', 
            'npv', 'roc', 'acc', 'prc']
    for col in cols:
        mu = np.array(result[f'{name}_{col}']).mean()
        std = np.array(result[f'{name}_{col}']).std()
        print(f'{col}: {mu*100:.0f} +- {200*std:.1f}')
        

def plot_risk_score_hist(scores, target, min_score, max_score, score_name, q_0, q_1, task="", filename="", savefig=False):
    """Utility function to plot: histogram of +/- scores, high-risk line at q_0, 
    low-risk line at q_1. Save figure under filename if savefig flag is used. 
    Args:
        scores (Int): Scores predicted.
        target (Int): True target that corresponds to each score.
        min_score (Int): Minimum possible score.
        max_score (Int): Maximum possible score.
        score_name (String): Name to identify score.
        q_0 (Int): Score that stratifies high-risk (i.e. > q_0)
        q_1 (Int): Score that stratifies low-risk (i.e. < q_1)
        task (String): Task/Dataset used to apply the scores to.
        filename (String): Filename to save plot to.
        savefig (Boolean): Indicator to save figure.
        
    Returns:
        (Object): Plt plot.
    """
    c_0 = scores[target==0]
    c_1 = scores[target==1]

    bins = np.arange(min_score - .5, max_score + 1.5, 1)
    plt.subplot(111, facecolor='#E6E6E6')
    plt.hist(c_0, bins, alpha=.5, label=f'Non-{task}')
    plt.hist(c_1, bins, alpha=.5, label=f'{task}')
    plt.title(f'Histograms of (n={N}) {score_name} scores')
    plt.axvline(x=q_1-.5, color = 'navy', linestyle = 'dashed')
    plt.axvline(x=q_0+.5, color = 'crimson', linestyle = 'dashed')
    plt.legend([f"Non-{task} ({len(c_0)})", f"{task} ({len(c_1)})", f"Low-Risk $<${q_1:.0f}", f"High-Risk $>${q_0:.0f}"], loc='upper right')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(axis = 'y')
    if savefig:
        plt.savefig(f"{filename}", dpi=300)
    return plt.show()

def get_single_exp(exp_id, directory='./logs'):
    """Helper function that returns a single dictionary with the parameters and results 
    logged by experiment with exp_id.

    Args:
        exp_id (str): Experiment ID.
        directory (str, optional): Directory to find exp_id. Defaults to './logs'.

    Returns:
        (Dictionary): 
    """
    f = os.path.join(directory, exp_id)
    try:
        with open(os.path.join(f, 'result.json')) as json_file:
            result = json.load(json_file)
        
        with open(os.path.join(f, 'params.json')) as json_file:
            params = json.load(json_file)
        
        params.update(result)
        return params
    except:
        return None
        
def generate_results_csv(filename, directory = './logs/benchmark'):
    """Load and save all experiment results into a single CSV.

    Args:
        filename (str): Filename to save csv into.
        directory (str, optional): Directory to load experiment results from. Defaults to './logs/benchmark'.
    
    Returns:
        None
    """
    ds = []
    for exp_id in os.listdir(directory):
        d = get_single_exp(exp_id, directory)
        if d is not None:
            ds += [d]
    pd.DataFrame(ds).to_csv(filename, index=False)
    
def eval_performance(y_true, y_pred, idx=None, y_prob=None, name=''):
    """Evaluate the prediction performance over a set of samples. 
    Args:
        y_true: Array of observed labels.
        y_pred: Array of predicted labels.
        idx: Boolean array of same size as y_true and y_pred.
        y_prob: Array of positive class probabilities.
    
    Returns:
        Dictionary with performance metrics.
    """
    result = {}
    
    result[f'{name}_coverage'] = 1 
    if idx is not None:
        result[f'{name}_coverage'] = np.sum(idx)/len(y_true)   
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    
    cm = metrics.confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    result[f'{name}_ppv'] = tp/(tp+fp)
    result[f'{name}_acc'] = (tp+tn)/(tn+fp+fn+tp)
    result[f'{name}_npv'] = tn/(tn+fn)
    result[f'{name}_sensitivity'] = tp/(tp+fn)
    result[f'{name}_specificity'] = tn/(tn+fp)
    result[f'{name}_specificity'] = tn/(tn+fp)
    result[f'{name}_prevalence'] = (tp+fn)/(tn+fp+fn+tp)
    result[f'{name}_f1'] = metrics.f1_score(y_true, y_pred)

    if y_prob is not None:
        if idx is not None:
            y_prob = y_prob[idx]
        fpr, tpr, auc_thresholds = metrics.roc_curve(y_true, y_prob)
        precision, recall, prc_thresholds = metrics.precision_recall_curve(y_true, y_prob)
        result[f'{name}_roc'] = metrics.auc(fpr, tpr)
        result[f'{name}_prc'] = metrics.auc(recall, precision)
        
    return result 

