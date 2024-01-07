from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.utils import resample
from joblib import Parallel, delayed
from sklearn.base import clone
import numpy as np
import itertools


def generate_param_grid(param_space, model_type=None):
    grid = []

    if model_type == 'logistic_regression':
        valid_combinations = {
            'lbfgs': ['l2', None],
            'liblinear': ['l1', 'l2'],
            'newton-cg': ['l2', None],
            'newton-cholesky': ['l2', None],
            'sag': ['l2', None],
            'saga': ['elasticnet', 'l1', 'l2', None]
        }

        for solver, penalty in itertools.product(param_space.get('solver', ['lbfgs']),
                                                 param_space.get('penalty', ['l2'])):
            if penalty in valid_combinations.get(solver, []):
                if penalty == 'elasticnet' and solver == 'saga':
                    for l1_ratio, C in itertools.product(param_space.get('l1_ratio', [None]),
                                                         param_space.get('C', [1.0])):
                        grid.append({'solver': solver, 'penalty': penalty, 'l1_ratio': l1_ratio, 'C': C})
                else:
                    for C in param_space.get('C', [1.0]):
                        grid.append({'solver': solver, 'penalty': penalty, 'C': C})

    elif model_type == 'svc':
        for kernel, C in itertools.product(param_space.get('kernel', ['rbf']),
                                           param_space.get('C', [1.0])):
            if kernel == 'poly':
                for degree, gamma in itertools.product(param_space.get('degree', [3]),
                                                       param_space.get('gamma', ['scale'])):
                    grid.append({'kernel': kernel, 'C': C, 'degree': degree, 'gamma': gamma})
            else:
                for gamma in param_space.get('gamma', ['scale']):
                    grid.append({'kernel': kernel, 'C': C, 'gamma': gamma})
    else:
        grid = list(ParameterGrid(param_space))

    return grid


def halving_grid_search(estimator, param_space, model, X, y, cv=3, n_iterations=3, scoring='accuracy', n_jobs=-1, factor=3):

    param_grid = generate_param_grid(param_space, model)
    n_candidates = len(param_grid)
    n_samples = len(X)
    best_score = -np.inf
    best_params = None
    history = []

    for iteration in range(n_iterations):

        iter_n_samples = min(n_samples, n_samples * (2 ** iteration) // (2 ** (n_iterations - 1)))
        X_iter, y_iter = resample(X, y, n_samples=iter_n_samples, replace=False)

        # Evaluate each combination of parameters
        scores = Parallel(n_jobs=n_jobs)(
            delayed(cross_val_score)(clone(estimator).set_params(**params), X_iter, y_iter, cv=cv, scoring=scoring)
            for params in param_grid
        )

        # Process and store the results
        for params, score in zip(param_grid, scores):
            mean_cv_score = np.mean(score)
            history.append((iteration, params, mean_cv_score, iter_n_samples))

        # Select the top candidates based on the factor
        mean_scores = np.mean(scores, axis=1)
        num_top_candidates = max(n_candidates // factor, 1)
        top_candidates = np.argsort(mean_scores)[-num_top_candidates:]
        param_grid = [param_grid[i] for i in top_candidates]

        # Update best score and parameters
        if mean_scores[top_candidates[-1]] > best_score:
            best_score = mean_scores[top_candidates[-1]]
            best_params = param_grid[-1]

        # Reduce the number of candidates for the next iteration
        n_candidates = len(top_candidates)

    return best_params, best_score, history


def hs_format_history(history):
    formatted_history = []
    for entry in history:
        iteration = entry['Iteration']
        params = entry['Parameters']
        mean_score = entry['Mean Test Score']
        resource_count = entry['Resource Count']

        formatted_entry = f"Iteration: {iteration}\nParameters: {params}\nScore: {mean_score:.6f}\Resource Count: {resource_count}\n"
        formatted_history.append(formatted_entry)
    return '\n'.join(formatted_history)