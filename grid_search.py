from sklearn.model_selection import ParameterGrid, cross_val_score
from tqdm import tqdm
import numpy as np
import time
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


def calculate_grid_size(model_type=None, param_grid=None):
    if model_type in ['logistic_regression', 'svc']:
        return len(param_grid)
    else:
        return len(list(param_grid))


def grid_search(model, param_grid, X, y, cv_times=3, scoring='accuracy', n_jobs=-1):
    best_score = -np.inf
    best_params = None
    history = []

    start_time = time.time()

    for i, params in enumerate(tqdm(param_grid, desc='Grid Search Progress')):
        model.set_params(**params)

        mean_cv_score = np.mean(cross_val_score(model, X, y, cv=cv_times, n_jobs=n_jobs))

        history.append((i, params, mean_cv_score))

        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_params = params

    end_time = time.time()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return best_params, best_score, history


def gs_format_history(history):
    formatted_history = []
    for entry in history:
        iteration = entry['Iteration']
        params = entry['Parameters']
        mean_score = entry['Mean Test Score']

        formatted_entry = f"Iteration: {iteration}\nParameters: {params}\nScore: {mean_score:.6f}\n"
        formatted_history.append(formatted_entry)

    return '\n'.join(formatted_history)