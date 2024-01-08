import streamlit as st
from customtool import smalltools


def generate_param_space_template(model_name):
    if model_name == 'Logistic Regression':
        return """import numpy as np
from sklearn.linear_model import LogisticRegression

param_space_lr = {
    'penalty': ['elasticnet', 'l1', 'l2', None],
    'C': [2**(-5), 2**(-3), 2**(-1), 2**1, 2**3, 2**5, 2**7, 2**9, 2**13],
    'solver': ['sag', 'saga', 'liblinear', 'lbfgs', 'newton-cg', 'newton-cholesky'],
    'l1_ratio': np.linspace(0, 1, 6)
}"""
    elif model_name == 'Support Vector Classifier':
        return """from sklearn.svm import SVC

param_space_svc = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [2**(-5), 2**(-3), 2**(-1), 1, 2**1, 5, 2**3, 10],
    'degree': [2, 3, 4, 5],  # Only for poly kernel
    'gamma': ['scale', 'auto']  # For rbf, poly, and sigmoid kernels
}"""
    elif model_name == 'Random Forest Classifier':
        return """from sklearn.ensemble import RandomForestClassifier

param_space_rfc = {
    'n_estimators': [80, 100, 150, 200, 250, 500],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [3, 4, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 3, 5, 7, 11, 17, 26, 40],
    'min_samples_leaf': [1, 2, 3, 4, 5, 7, 10, 15, 20],
    'max_features': ['sqrt', 'log2', None]
}"""
    elif model_name == 'Gradient Boosting Classifier':
        return """from sklearn.ensemble import GradientBoostingClassifier

param_space_gbc = {
    'loss': ['log_loss', 'exponential'],
    'learning_rate': [0.001, 0.00215443, 0.00464159, 0.01, 0.02154435, 0.04641589, 0.1, 0.21544347, 0.46415888, 1.],
    'n_estimators': [80, 100, 150, 200, 250, 500],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2, 3, 4, 5, 7, 10, 15, 20],
    'max_depth': [3, 4, 5, 7, 10, 15, 20],
    'max_features': ['sqrt', 'log2', None]
}"""


def model_selection_form():
    model_names = ['Logistic Regression', 'Support Vector Classifier',
                   'Random Forest Classifier', 'Gradient Boosting Classifier']
    model_name = st.selectbox('Choose a model', model_names)
    param_space_code = generate_param_space_template(model_name)
    st.write("### ")
    st.write("Copy the code and modify the parameter space as needed in your environment.")
    return model_name, param_space_code


def generate_scoring_code(selected_metrics, average_method):

    import_statements = "from sklearn.metrics import make_scorer"

    metric_to_function = {
        'accuracy': 'accuracy_score',
        'precision': 'precision_score',
        'recall': 'recall_score',
        'f1_score': 'f1_score',
        'roc_auc': 'roc_auc_score'
    }

    for metric in selected_metrics:
        if metric in metric_to_function:
            import_statements += f", {metric_to_function[metric]}"

    # Generate scoring code
    code = f"{import_statements}\n\ntrain_scoring = {{\n"
    for metric in selected_metrics:
        if metric in ['precision', 'recall', 'f1_score', 'roc_auc']:
            code += f"    '{metric}': make_scorer({metric_to_function[metric]}, average='{average_method}'),\n"
        else:
            code += f"    '{metric}': make_scorer({metric_to_function[metric]}),\n"
    code += "}"
    return code


def user_input_scoring():
    with st.form(key='scoring_form'):
        # Selection for scoring metrics
        selected_metrics = st.multiselect('Select Scoring Metrics',
                                          ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
                                          default=['accuracy'])

        # Average options for precision, recall, f1_score, and roc_auc
        average_method = st.selectbox("Select averaging method for precision, recall, f1_score, and roc_auc",
                                      ('macro', 'micro', 'weighted', 'samples'), index=0)

        st.markdown("""
        **Note on Average Method:**
        - 'macro': Calculate metrics for each label, and find their unweighted mean. Does not take label imbalance into account.
        - 'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
        - 'weighted': Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
        - 'samples': Calculate metrics for each instance, and find their average.
        """)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.session_state['selected_metrics'], _ = selected_metrics, average_method
            return selected_metrics, average_method
        else:
            return [], 'macro'


def user_input_parameters():
    with st.form(key='param_space_form'):
        cv_times = st.number_input('Number of Cross-Validation Splits', min_value=2, max_value=20, value=5, step=1)
        seed_value = st.number_input('Seed Value for Random State', value=42, step=1)
        n_jobs = st.number_input('Number of Jobs to Run in Parallel (-1 for all CPUs)', min_value=-1, value=-1, step=1)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            param_str = f"cv_times={cv_times}\nseed_value={seed_value}\nn_jobs={n_jobs}"
            st.code(param_str, language='python')
            return {
                'cv_times': cv_times,
                'seed_value': seed_value,
                'n_jobs': n_jobs
            }


def generate_template_code(model_name, method, params, selected_metrics):
    model_info = {
        'Logistic Regression': {'class': 'LogisticRegression', 'abbr': 'lr', 'model_type': 'logistic_regression'},
        'Support Vector Classifier': {'class': 'SVC', 'abbr': 'svc', 'model_type': 'svc'},
        'Random Forest Classifier': {'class': 'RandomForestClassifier', 'abbr': 'rfc', 'model_type': 'rfc'},
        'Gradient Boosting Classifier': {'class': 'GradientBoostingClassifier', 'abbr': 'gbc', 'model_type': 'gbc'}
    }

    metric_print_statements = '\n    '.join([
        f'print("Best {metric.title().replace("_", " ")}:", {model_info[model_name]["abbr"]}_train_result["mean_{metric}"])'
        for metric in selected_metrics
    ])

    method_templates = {
        'Grid Search': '''
    from grid_search import calculate_grid_size, generate_param_grid, grid_search, gs_format_history
    from plot_history import plot_score_vs_iterations
    
    param_grid_lr = generate_param_grid(param_space_{abbr}, model_type='{model_type}')
    print('Param grid size: ', calculate_grid_size(param_grid=param_grid_{abbr}, model_type='{model_type}'))
    
    {model_variable} = {model_class}(random_state={seed_value})  
    {abbr}_gs_best_params, {abbr}_gs_train_result, {abbr}_gs_history, {abbr}_gs_time = grid_search(model={model_variable}, param_grid=param_grid_{abbr}, 
                                                                                   scoring=train_scoring, X=X_train, y=y_train, 
                                                                                   cv_times={cv_times}, n_jobs={n_jobs}, seed={seed_value},
                                                                                   parallelize=True)

    print("Best Parameters:", {abbr}_gs_best_params)
    {metric_print_statements}
    print("Total time taken:", {abbr}_gs_time, "seconds")
    
    plot_score_vs_iterations({abbr}_gs_history)
    print(gs_format_history({abbr}_gs_history))
    ''',
        'Simulated Annealing':'''
    from simulated_annealing import simulated_annealing, sa_format_history
    from plot_history import plot_score_vs_iterations
    
    {model_variable} = {model_class}(random_state={seed_value})
    {abbr}_sa_best_params, {abbr}_sa_train_result, {abbr}_sa_history, {abbr}_sa_time = simulated_annealing(model={model_variable}, 
                                                                                            param_space=param_space_{abbr}, 
                                                                                            max_iter=50, scoring=train_scoring, 
                                                                                            X=X_train, y=y_train, 
                                                                                            initial_temp=100, cooling_rate=0.90, cv_times={cv_times}, 
                                                                                            model_type='{model_type}',
                                                                                            n_jobs={n_jobs}, seed={seed_value})

    print("Best Parameters:", {abbr}_sa_best_params)
    {metric_print_statements}
    print("Total time taken:", {abbr}_sa_time, "seconds")
    
    plot_score_vs_iterations({abbr}_sa_history)
    print(sa_format_history({abbr}_sa_history))
    ''',
        'Half Grid Search':'''
    from half_grid_search import halving_grid_search, hs_format_history
    from plot_history import plot_score_vs_iterations
    
    {model_variable} = {model_class}(random_state={seed_value})
    {abbr}_hs_best_params, {abbr}_hs_train_result, {abbr}_hs_history, {abbr}_hs_time = halving_grid_search(model={model_variable}, 
                                                                                           param_space=param_space_{abbr}, 
                                                                                           scoring=train_scoring, 
                                                                                           X=X_train, y=y_train, 
                                                                                           cv_times={cv_times}, n_iterations=10, factor=2,
                                                                                           dynamic_reduction=False, n_jobs={n_jobs},
                                                                                           model_type='{model_type}',
                                                                                           seed={seed_value})

    print("Best Parameters:", {abbr}_hs_best_params)
    {metric_print_statements}
    print("Total time taken:", {abbr}_hs_time, "seconds")
    
    plot_score_vs_iterations({abbr}_hs_history) 
    print(hs_format_history({abbr}_hs_history))   
    ''',
        'Genetic Algorithm':'''
    from genetic_algorithm import genetic_algorithm, format_ga_history, access_population_info
    from plot_history import plot_score_vs_iterations
    
    {model_variable} = {model_class}(random_state={seed_value})
    {abbr}_ga_best_params, {abbr}_ga_train_result, {abbr}_ga_history, {abbr}_ga_time = genetic_algorithm(model={abbr}_model_ga, 
                                                                                         param_space=param_space_{abbr}, 
                                                                                         scoring=train_scoring, X=X_train, 
                                                                                         y=y_train, cv_times={cv_times}, 
                                                                                         pop_size=5, max_iterations=10, 
                                                                                         no_improve_limit=100, n_jobs={n_jobs}, 
                                                                                         sel_rate=0.2, mut_rate=0.8, 
                                                                                         model_type='{model_type}',
                                                                                         seed={seed_value})

    print("Best Parameters:", {abbr}_ga_best_params)
    {metric_print_statements}
    print("Total time taken:", {abbr}_ga_time, "seconds")
    
    plot_score_vs_iterations({abbr}_ga_history)
    print(format_ga_history({abbr}_ga_history))
    access_population_info({abbr}_ga_history)
    '''}

    model_class = model_info.get(model_name, {'class': 'UnknownModel', 'abbr': 'unknown', 'model_type': 'unknown'}).get('class')
    model_variable = model_class.split()[0].lower()
    model_type = model_info.get(model_name, {'class': 'UnknownModel', 'abbr': 'unknown', 'model_type': 'unknown'}).get('model_type')
    abbr = model_info.get(model_name, {'class': 'UnknownModel', 'abbr': 'unknown', 'model_type': 'unknown'}).get('abbr')

    method_template = method_templates.get(method, 'Method not available for this model')
    template_code = method_template.format(model_class=model_class, abbr=abbr, model_variable=model_variable, model_type=model_type,
                                           seed_value=params['seed_value'], n_jobs=params['n_jobs'], cv_times=params['cv_times'],
                                           metric_print_statements=metric_print_statements)
    return template_code


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Create Hyperparameter Tuning Code")
    st.write('This section will help you create code to tune the model instantly. You have to select the given options, copy '
             'and paste the given code to your environment to run it. You can modify the given templates, codes used in '
             'hyperparameter tuning or anything else as long as it fulfills your need. Enjoy! 🤗')

    if 'selected_metrics' not in st.session_state:
        st.session_state['selected_metrics'] = []

    if 'user_input_parameters' not in st.session_state:
        st.session_state['user_input_parameters'] = {'cv_times': 5, 'seed_value': 42, 'n_jobs': -1}

    st.divider()
    st.markdown("## Step 0 - Download Necessary Files")
    st.write('Please download `.py file` for hyperparameter tuning method and `plot_history.py` in GitHub. '
             'Click this button to direct you to the website.')
    st.link_button(label='GitHub Portal', url='https://github.com/ryoshi007/Hyperparameter-Tuning')

    st.divider()
    st.markdown("## Step 1 - Create Parameter Space")
    st.write('Select your desirable model and copy the template provided for setting up parameter space.')
    selected_model, user_code = model_selection_form()

    if user_code:
        st.markdown("#### Parameter Space Code:")
        st.code(user_code, language='python')

    st.divider()
    st.markdown("## Step 2 - Define Scoring Metric")
    st.write('Define the scoring metric for the model training. This will be used to evaluate the model during hyperparameter tuning.')

    selected_metrics, average_method = user_input_scoring()

    if selected_metrics:
        st.write('Copy and paste this code in your environment.')
        scoring_code = generate_scoring_code(selected_metrics, average_method)
        st.code(scoring_code, language='python')

    st.divider()
    st.markdown("## Step 3 - Additional Parameters")
    st.write('Choose values for number of cross validations, seed value and number of jobs.')
    additional_params = user_input_parameters()

    if additional_params:
        st.session_state['user_input_parameters'] = additional_params

    st.divider()
    st.markdown("## Step 4 - Choose Hyperparameter Tuning Method")

    # Selection for hyperparameter tuning method
    methods = ['Grid Search', 'Half Grid Search', 'Simulated Annealing', 'Genetic Algorithm']
    selected_method = st.selectbox('Choose a hyperparameter tuning method', methods, key='method_select')

    if st.button('Generate Template Code', key='generate_code'):
        st.write('Please adjust the specified parameters (initial_temperature, budget, pop_size, ...) based on your usage:')
        template_code = generate_template_code(selected_model, selected_method, st.session_state['user_input_parameters'], st.session_state['selected_metrics'])
        st.code(template_code, language='python')

    st.divider()
    st.markdown("## Step 5 - Run the Code")
    st.write('After copied and pasted all the given codes, you should be able to run the codes without any issues. ✨')


load_page()
