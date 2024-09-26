import click
import json
import os
import time
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from utils import prepare_experiment, aggregate_results, eval_performance
from models.FasterRisk import FasterRisk, grid_search

@click.command()
@click.option(
    "--dataset",
    default="electricity",
    help="Identifier name of dataset."
)
@click.option(
    "--category",
    default="clf_cat",
    help="Category for HuggingFace tabular datasets"
)
@click.option(
    "--log_path",
    default="./logs/benchmark",
    help="Path to log experiments."
)
@click.option(
    "--seed",
    default=1234,
    help="Random seed.",
)
@click.option(
    "--nfolds",
    default=2,
    help="Number of folds"
)
@click.option(
    "--val_size", 
    default=.2,
    help="Val size to split train data"
)
@click.option(
    "--n_trials",
    default=1,
    help="Number of trials for grid search"
)
@click.option(
    "--b_fpr",
    default=.1,
    help="Upper bound for FPR."
)
@click.option(
    "--b_fnr",
    default=.1,
    help="Upper bound for FNR."
)
    
@click.pass_context
def main(ctx, dataset, category, log_path, seed, nfolds, val_size, n_trials, b_fpr, b_fnr):
    # Load and preprocess data
    data = load_dataset("inria-soda/tabular-benchmark", data_files=f"{category}/{dataset}.csv")
    df = data['train'].to_pandas()
    df = preprocess(df, dataset, category)
    
    internal_data = df.drop('target', axis=1)
    internal_target = df['target']
    feats = [f"X{i}" for i in range(1, df.shape[1])]

    # Setup experiment log directory.
    exp_id, exp_path = prepare_experiment(log_path)
    
    # Setup start time.
    start = time.time()

    error = [b_fpr, b_fnr]
    results = []
    skf = StratifiedKFold(n_splits=nfolds)
    for i, (train_index, test_index) in enumerate(skf.split(internal_data, internal_target)):
        print(f"Starting fold {i}")

        # Split data into train and test
        X_train = internal_data.iloc[train_index].to_numpy().astype(np.int32)
        y_train = internal_target.iloc[train_index].to_numpy().astype(np.int32)
        X_test = internal_data.iloc[test_index].to_numpy().astype(np.int32)
        y_test = internal_target.iloc[test_index].to_numpy().astype(np.int32)

        # Get optimal hyperparameters.
        X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)
        hyperparams = grid_search(X, y, X_val, y_val, feats, n_trials)

        # Estimate model
        model = FasterRisk(**hyperparams)
        model.fit(X_train, y_train, feats)
        
        result={}
        # Evaluate model on internal cohort
        y_pred = model.model.predict(X_test)
        y_pred = binarize(y_pred)
        y_pred_prob = model(X_test)
    
        # Evaluate model on external cohort
        result.update(eval_performance(y_test, binarize(model.model.predict(X_test)) , None, y_prob=model(X_test), name='FasterRisk'))
        results += [result]
        q_0, q_1 = CC(model.score(X_val),binarize(y_val), error)
        FR_test_scores = model.score(X_test)
        idx_conformal = np.logical_xor(FR_test_scores<q_1, FR_test_scores>q_0)
        conformal_pred =  (FR_test_scores>=q_1).astype(np.int32) 
        result.update(eval_performance(y_test, conformal_pred, idx_conformal, y_prob=model(X_test), name='FasterRisk+CC'))

        # Compute metrics
        cm = confusion_matrix(y_test[idx_conformal], conformal_pred[idx_conformal])
        tn, fp, fn, tp = cm.ravel()
        n = tp+fp+tn+fn
        P_ACS = (y_test[idx_conformal]==1).mean()

        N = len(y_test)
        test_N0 = (y_test==0).sum()
        test_N1 = (y_test==1).sum()

        print(f" |^P |^N |\nP|{tp:3d}|{fn:3d}|\nN|{fp:3d}|{tn:3d}|")
        print(f"Sensitivity: {tp/(tp+fn):.2f} | NPV:{tn/(tn+fn):.2f}")
        print(f"E Coverage:{idx_conformal.sum()/len(y_test):.2f} | T Coverage:{(1-(((model.score(X_test)>=q_1)&(model.score(X_test)<=q_0)).sum()/len(model.score(X_test)))):.2f}")
        print(f"E Error:{((fn+fp)/N):.3f} | T Error Upper Bound:{error[0]*(1-P_ACS)+error[1]*(P_ACS):.3f}")
        print(f"E FP:{fp/(test_N0):.3f} | T FP:{error[0]:.3f}")
        print(f"E FN:{fn/(test_N1):.3f} | T FN:{error[1]:.3f}")
        print(f"E NPV:{tn/(tn+fn)}")
        print(f"E PPV:{tp/(tp+fp)}")
       
        result.update({'E FPR':fp/(test_N0),
                       'E FNR':fn/(test_N1),
                       'E Error':(fn+fp)/N,
                       'T FPR':error[0],
                       'T FNR':error[1],
                       'T Error':(error[0]*(1-P_ACS))+(error[1]*(P_ACS))})
    
    end = time.time()
    result = aggregate_results(results)
    result['exp_id']=str(int(exp_id))
    result['time']=end - start

    # Save result dict
    with open(os.path.join(exp_path, 'result.json'), "w") as outfile:
        json.dump(result, outfile)

    with open(os.path.join(exp_path, 'params.json'), "w") as outfile:
        json.dump(ctx.params, outfile)


def preprocess(df, dataset_name, category):
    """Preprocess tabular dataset.

    Args:
        df (_type_): _description_
        dataset_name (_type_): _description_
        category (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    with open('src/config.json', 'r') as file:
        config = json.load(file)
    
    category_config = config.get(category, {})
    target_info = category_config.get(dataset_name)
    
    if target_info is None:
        raise ValueError(f"No configuration found for dataset {dataset_name}")
    
    target_column = target_info['target_column']
    mapping = target_info['mapping']

    if mapping:
        df[target_column] = df[target_column].astype(str)
        df['target'] = df[target_column].map(mapping)
    else:
        df['target'] = df[target_column]
    
    df = df.drop(columns=[target_column])

    return df
    

def binarize(x):
    """Transform x to binary

    Args:
        x (float): Array of floats.

    Returns:
        bool: Binary value of x
    """
    x[x<0]=0
    return x


def CC(scores, target, error=[.1, .1]):
    """Class conditional conformal estimation algorithm.

    Args:
        scores (float): Array of scores.
        target (int): Array of binary targets.
        error (list, optional): Upper bound on FPR and FNR respectively. Defaults to [.1, .1].
    """
    N0 = len(scores[target==0])
    X0 = scores[target==0]
    q_0 = np.sort(X0)[int(np.ceil((N0+1)*(1-error[0])))]

    N1 = len(scores[target==1])
    X1 = scores[target==1] 
    q_1 = np.sort(X1)[int(np.floor((N1+1)*(error[1])))]
    return q_0, q_1


if __name__ == "__main__":
    main()
