import numpy as np
import random
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import time


def objective_function(model, params, X, y, cv_times, n_jobs=-1):
    model.set_params(**params)
    return np.mean(cross_val_score(model, X, y, cv=cv_times, n_jobs=n_jobs))


def get_random_params(param_space, current_params, model_type):
    next_params = current_params.copy()

    if model_type == 'logistic_regression':
        # Logistic Regression specific logic
        next_params['solver'] = random.choice(param_space.get('solver', [None]))
        if 'penalty' in param_space:
            penalty_options = {
                'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
                'newton-cg': ['l2', None], 'newton-cholesky': ['l2', None],
                'sag': ['l2', None], 'saga': ['elasticnet', 'l1', 'l2', None]
            }
            next_params['penalty'] = random.choice(penalty_options[next_params['solver']])

            if next_params['penalty'] == 'elasticnet' and next_params['solver'] == 'saga':
                next_params['l1_ratio'] = random.choice(param_space.get('l1_ratio', [None]))
            else:
                next_params.pop('l1_ratio', None)
            if next_params['penalty'] is not None:
                next_params['C'] = random.choice(param_space.get('C', [1.0]))
            else:
                next_params.pop('C', None)

    elif model_type == 'svc':
        # SVC specific logic
        next_params['kernel'] = random.choice(param_space.get('kernel', [None]))
        if next_params['kernel'] == 'poly':
            next_params['degree'] = random.choice(param_space.get('degree', [None]))
        if next_params['kernel'] in ['rbf', 'poly', 'sigmoid']:
            next_params['gamma'] = random.choice(param_space.get('gamma', [None]))

    # Update other parameters common to all models
    for key in param_space:
        if model_type == 'logistic_regression' and key in ['solver', 'penalty', 'l1_ratio', 'C']:
            continue
        if model_type == 'svc' and key in ['kernel', 'degree', 'gamma']:
            continue
        next_params[key] = random.choice(param_space[key])

    return next_params


def simulated_annealing(model, param_space, X, y, model_type, n_jobs, max_iter=100, initial_temp=100, cooling_rate=0.95,
                        cv_times=3):
    initial_params = {k: random.choice(v) for k, v in param_space.items()}
    current_params = get_random_params(param_space, initial_params, model_type)
    current_score = objective_function(model, current_params, X, y, cv_times, n_jobs)
    best_params = current_params
    best_score = current_score

    history = [(0, current_params, current_score, 'Initial configuration', 0, initial_temp)]
    temp = initial_temp

    start_time = time.time()
    for i in tqdm(range(1, max_iter + 1), desc='Simulated Annealing Progress'):
        next_params = get_random_params(param_space, current_params, model_type)
        next_score = objective_function(model, next_params, X, y, cv_times, n_jobs)

        diff = next_score - current_score
        prob = np.exp(diff / temp)

        if diff > 0 or random.uniform(0, 1) < prob:
            current_params, current_score = next_params, next_score
            if current_score > best_score:
                best_params, best_score = current_params, current_score
                history.append((i, next_params, next_score, 'Improvement and accepted', diff, temp))
            else:
                history.append((i, next_params, next_score, 'No improvement and accepted', diff, temp))
        else:
            history.append((i, next_params, next_score, 'No improvement and rejected', diff, temp))

        temp *= cooling_rate

    elapsed_time = time.time() - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    return best_params, best_score, history


def sa_format_history(history):
    formatted_history = []
    for entry in history:
        iteration, params, score, status, diff, temp = entry
        formatted_entry = f"Iteration: {iteration}\nParameters: {params}\nScore: {score:.6f}\nStatus: {status}\nScore Difference: {diff:.6f}\nTemp: {temp}\n"
        formatted_history.append(formatted_entry)
    return '\n'.join(formatted_history)
