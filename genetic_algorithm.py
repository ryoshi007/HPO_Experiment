import numpy as np
import random
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


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


def validate_params(params, param_space, model_type):
    valid_params = params.copy()

    if model_type == 'logistic_regression':
        # Adjust for logistic regression constraints
        solver = valid_params.get('solver', None)
        penalty_options = {
            'lbfgs': ['l2', None], 'liblinear': ['l1', 'l2'],
            'newton-cg': ['l2', None], 'newton-cholesky': ['l2', None],
            'sag': ['l2', None], 'saga': ['elasticnet', 'l1', 'l2', None]
        }

        if solver in penalty_options:
            valid_penalty = penalty_options[solver]
            if valid_params.get('penalty') not in valid_penalty:
                valid_params['penalty'] = random.choice(valid_penalty)

            if valid_params['penalty'] == 'elasticnet' and solver == 'saga':
                # Update l1_ratio only if it's not already set or if it's invalid
                if 'l1_ratio' not in valid_params or valid_params['l1_ratio'] not in param_space.get('l1_ratio', []):
                    valid_params['l1_ratio'] = random.choice(param_space.get('l1_ratio', [None]))
            else:
                valid_params.pop('l1_ratio', None)

            if valid_params['penalty'] is not None:
                # Update C only if it's not already set or if it's invalid
                if 'C' not in valid_params or valid_params['C'] not in param_space.get('C', []):
                    valid_params['C'] = random.choice(param_space.get('C', [1.0]))
            else:
                valid_params.pop('C', None)

        # SVC adjustments
        elif model_type == 'svc':
            kernel = valid_params.get('kernel', None)

            if kernel == 'poly':
                # Update 'degree' only if not set or invalid
                if 'degree' not in valid_params or valid_params['degree'] not in param_space.get('degree', []):
                    valid_params['degree'] = random.choice(param_space.get('degree', [None]))
            else:
                valid_params.pop('degree', None)

            if kernel in ['rbf', 'poly', 'sigmoid']:
                # Update 'gamma' only if not set or invalid
                if 'gamma' not in valid_params or valid_params['gamma'] not in param_space.get('gamma', []):
                    valid_params['gamma'] = random.choice(param_space.get('gamma', [None]))
            else:
                valid_params.pop('gamma', None)

    return valid_params


def crossover(parent1, parent2):
    child = {}
    for param in parent1:
        child[param] = parent1[param] if random.random() < 0.5 else parent2[param]
    return child


def mutate(chromosome, mutation_rate, param_space, model_type):
    if random.random() < mutation_rate:
        return get_random_params(param_space, chromosome, model_type)
    return chromosome


def create_initial_population(pop_size, param_spaces, model_types):
    population = []
    for _ in range(pop_size):
        model_type = random.choice(model_types)
        params = get_random_params(param_spaces[model_type], {}, model_type)
        population.append((model_type, params))
    return population


def update_rates(fit_avg, fit_max, selrate, mutrate):
    ratio = fit_avg / fit_max
    selrate += 0.1 * ratio
    mutrate -= 0.1 * ratio
    return selrate, mutrate


def genetic_algorithm(X, y, param_spaces, model, model_types, pop_size, max_iterations, no_improve_limit, sel_rate,
                      mut_rate, cv_times=3, n_jobs=-1):
    # Initialize population
    population = create_initial_population(pop_size, param_spaces, model_types)
    history = []

    best_score = float('-inf')
    best_params = None
    no_improve_rounds = 0

    for iteration in tqdm(range(max_iterations), desc="GA Progress"):
        # Evaluate population
        fitness = [objective_function(model, params, X, y, cv_times, n_jobs) for model_type, params in
                   population]

        # Check for improvement
        max_fitness = max(fitness)
        if max_fitness > best_score + 0.001:
            best_score = max_fitness
            best_params = population[fitness.index(max_fitness)][1]
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= no_improve_limit:
                break

        # Update adaptive rates
        avg_fitness = np.mean(fitness)
        sel_rate, mut_rate = update_rates(avg_fitness, max_fitness, sel_rate, mut_rate)

        # Selection (random selection of parents for now)
        parents = random.sample(population, k=int(sel_rate * pop_size))

        # Crossover and mutation
        offspring = []
        for _ in range(pop_size - len(parents)):
            parent1, parent2 = random.sample(parents, 2)
            child_params = crossover(parent1[1], parent2[1])
            child_params = mutate(child_params, mut_rate, param_spaces[parent1[0]], parent1[0])
            child_params = validate_params(child_params, param_spaces[parent1[0]], parent1[0])
            offspring.append((parent1[0], child_params))

        # Create new population
        population = parents + offspring
        history.append((iteration, best_params, best_score, avg_fitness, sel_rate, mut_rate))

    return best_params, best_score, history


def format_ga_history(history):
    formatted_history = []
    for entry in history:
        iteration, best_params, max_fitness, avg_fitness, sel_rate, mut_rate = entry

        formatted_entry = (
            f"Iteration: {iteration}\n"
            f"Best Parameters: {best_params}\n"
            f"Max Fitness Score: {max_fitness:.6f}\n"
            f"Average Fitness Score: {avg_fitness:.6f}\n"
            f"Selection Rate: {sel_rate:.6f}\n"
            f"Mutation Rate: {mut_rate:.6f}\n"
        )
        formatted_history.append(formatted_entry)
    return '\n'.join(formatted_history)