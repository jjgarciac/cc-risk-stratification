sys.path.append(".") 
import keras
import numpy as np
import optuna
import sys
from src.utils import eval_performance
from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier

class FasterRisk(keras.Model):
    def __init__(self, 
                 sparsity=10, 
                 gap_tolerance=0.3, 
                 lb=-5, 
                 select_top_m=50, 
                 ub=5, 
                 parent_size=10, 
                 child_size=None, 
                 maxAttempts=50, 
                 num_ray_search=20, 
                 lineSearch_early_stop_tolerance=0.001):
        
        super(FasterRisk, self).__init__()
        self.optimizer = None
        self.model = None
        self.sparsity = sparsity
        self.gap_tolerance = gap_tolerance
        self.lb = lb
        self.select_top_m = select_top_m
        self.ub = ub
        self.parent_size = parent_size
        self.child_size = child_size
        self.maxAttempts = maxAttempts
        self.num_ray_search = num_ray_search
        self.lineSearch_early_stop_tolerance = lineSearch_early_stop_tolerance

    def call(self, x):
        return self.model.predict_prob(x.numpy())

    def fit(self, X_train, y_train, featureNames):
        self.optimizer = RiskScoreOptimizer(X = X_train, y = (y_train * 2 - 1), k = self.sparsity, 
                                            gap_tolerance=self.gap_tolerance, lb=self.lb, 
                                            select_top_m=self.select_top_m, ub=self.ub)
        self.optimizer.optimize()
        multiplier, intercept, coefficients = self.optimizer.get_models(model_index = 0) 
        self.model = RiskScoreClassifier(multiplier = multiplier, intercept = intercept, 
                                         coefficients = coefficients, featureNames = featureNames)
        
    def predict(self, x, lr = -1, hr = 1):
        y = self.score(x)
        predictions = np.zeros_like(y, dtype=float)
        predictions[(y > lr) & (y < hr)] = 0.5
        predictions[y >= hr] = 1
        return predictions
    
    def score (self, x):
        return x.dot(self.model.coefficients)
    
    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)  
    
def grid_search(X_train, y_train, X_val, y_val, featureNames, n_trials=100, metric='acc'):
    def objective(trial):
        sparsity = trial.suggest_categorical('sparsity', [10, 20, 15, 5])
        gap_tolerance = trial.suggest_float('gap_tolerance', 0.3, 0.4)
        lb = trial.suggest_categorical('lb', [-10, -5])
        select_top_m = trial.suggest_categorical('select_top_m', [50])
        ub = trial.suggest_categorical('ub', [10, 5])
        parent_size = trial.suggest_categorical('parent_size', [20])
        child_size = trial.suggest_categorical('child_size', [20])
        maxAttempts = trial.suggest_categorical('maxAttempts', [50])
        num_ray_search = trial.suggest_categorical('num_ray_search', [20])
        lineSearch_early_stop_tolerance = trial.suggest_categorical('lineSearch_early_stop_tolerance', [0.001])


        m = FasterRisk(sparsity=sparsity, gap_tolerance=gap_tolerance, lb=lb, select_top_m=select_top_m, ub=ub, 
                       parent_size=parent_size, child_size=child_size, maxAttempts=maxAttempts, 
                       num_ray_search=num_ray_search, lineSearch_early_stop_tolerance=lineSearch_early_stop_tolerance)

        m.fit(X_train, y_train, featureNames=featureNames)

        def binarize(x):
            x[x<0]=0
            return x

        y_pred = binarize(m.model.predict(X_val))
        y_prob = m(X_val)
        result = eval_performance(y_val, y_pred, None, y_prob, name="gs")
        return result[f'gs_{metric}']
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return study.best_params
