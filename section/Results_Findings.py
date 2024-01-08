import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
from streamlit_extras.dataframe_explorer import dataframe_explorer
from st_clickable_images import clickable_images
import pandas as pd
import warnings
import base64


@st.cache_resource
def load_dataset(file_path):
    return pd.read_csv(file_path)


def computation_time():
    st.write("### ")
    st.write('### Average Completion Time per Seed')
    st.write("##### ")
    col1, col2, col3 = st.columns(spec=[0.1,0.8,0.1])
    with col2:
        st.image('static/Average Completion Time.png', use_column_width=True)
        st.markdown("""
        **Grid Search** took the **longest** time to complete, while **Genetic Algorithm** used the **least** amount of time
        """)

    st.write("### ")
    st.divider()

    st.write('### Average Completion Time for Hyperparameter Tuning by ML Model')
    st.write("##### ")
    col1, col2, col3 = st.columns(spec=[0.1,0.8,0.1])
    with col2:
        st.image('static/Hyperparameter By ML Model.png', use_column_width=True)
        st.markdown("""
        **Completion Time Rank**

        1. Genetic Algorithm
        2. Simulated Annealing
        3. Half Grid Search
        4. Grid Search
        """)

    st.write("### ")
    st.divider()

    st.write('### Completion Time Data')
    data = load_dataset('data/comparison_time.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)


def quality_assessment():
    st.write("### ")

    st.write('### Frequency Distribution of Hyperparameter Tuning Methods Ranked in Top 2 for ROC AUC Improvement')
    st.write('Select each picture to get its explanation.')
    st.write("##### ")
    images = []
    for file in ['static/Frequency of Hyperparameter Tuning Methods in Top 1.png',
                 'static/Frequency of Hyperparameter Tuning Methods in Top 2.png',
                 'static/Frequency of Hyperparameter Tuning Methods in Top 3.png',
                 'static/Frequency of Hyperparameter Tuning Methods in Top 4.png']:
        with open(file, 'rb') as image:
            encoded = base64.b64encode(image.read()).decode()
            images.append(f'data:image/png;base64, {encoded}')
    clicked = clickable_images(
        images,
        titles=[f"Image #{str(i)}" for i in range(len(images))],
        div_style={"display": "flex", "flex-direction": "row", "justify-content": "center", "flex-wrap": "wrap",
                   "align-items": "center"},
        img_style={"margin": "5px", "width": "calc(50% - 10px)", 'height': 'auto'},
    )

    if int(clicked) == 0:
        st.write('ROC AUC score is used due to imbalanced testing set')
    elif int(clicked) == 1:
        st.markdown("""
        **Baseline**: Effective for **simple** models   
**Genetic Algorithm**: Highly effective for **complex** models
        """)
    elif int(clicked) == 2:
        st.markdown("""
            **Grid Search**: Perform **averagely** across all cases  
    **Half Grid Search**: Show improved quality as model **complexity increases**
            """)
    elif int(clicked) == 3:
        st.markdown("""
            **Simulated Annealing**: Well-suited for models of moderate complexity
            """)

    st.write("### ")
    st.divider()

    st.write('### Average Improvement by Hyperparameter Tuning Methods')
    st.write("##### ")
    col2, col3 = st.columns(spec=[0.5, 0.5], gap='small')
    with col2:
        st.image('static/Average Improvement by Hyperparameter Tuning Methods 1.png', use_column_width=True)
        st.markdown("""
**Logistic Regression** and **Random Forest Classifier** receive **less** benefit after optimization
        """)

    with col3:
        st.image('static/Average Improvement by Hyperparameter Tuning Methods 2.png', use_column_width=True)
        st.markdown("""
**Support Vector Classifier** and **Gradient Boosting Classifier** receive **around 1-2% of improvement** after optimization
        """)

    st.write("### ")
    st.divider()

    st.write('### Best ROC AUC Data')
    data = load_dataset('data/best_roc_auc_data.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)

    st.write("### ")
    st.divider()

    st.write('### Average Improvement Data')
    data = load_dataset('data/average_improvement_data.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)


def exploratory():
    st.write("### ")
    st.write('### Absolute Manhattan Distance Variations across Tuning Methods')
    st.write("##### ")
    col1, col2, col3 = st.columns(spec=[0.1,0.8,0.1])
    with col2:
        st.image('static/Absolute Manhattan Distance Variations across Tuning Methods.png', use_column_width=True)
        st.markdown("""
**Grid Search** has an obvious **smaller** spread of Manhattan distance across every model
        """)

    st.write("### ")
    st.divider()
    st.write('### Average Manhattan Distance Variations across Tuning Methods')
    st.write("##### ")
    col2, col3 = st.columns(spec=[0.5, 0.5], gap='small')
    with col2:
        st.image('static/Average Manhattan Distance across Tuning Methods 1.png', use_column_width=True)
        st.markdown("""
**Higher** average distance variation -> Have **more** different sets of solutions  
Have **more** different sets of solutions -> **Greater** exploratory capability   
        """)

    with col3:
        st.image('static/Average Manhattan Distance across Tuning Methods 2.png', use_column_width=True)
        st.markdown("""
**Total Average Manhattan Distance Variations Rank**

1. Genetic Algorithm
2. Simulated Annealing
3. Half Grid Search
4.Grid Search
        """)

    st.write("### ")
    st.divider()

    st.write('### Pairwise Distance between Hyperparameter Solution Points Data')
    data = load_dataset('data/pairwise_distance.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)

    st.write("### ")
    st.divider()

    st.write('### Average Manhattan Distance Variations Data')
    data = load_dataset('data/average_manhattan_distance.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)


def model_performance():
    st.write("### ")
    st.write('### Baseline Model Performance')
    st.write("##### ")
    col1, col2, col3 = st.columns(spec=[0.1,0.8,0.1])
    with col2:
        st.image('static/Baseline Model Performance.png', use_column_width=True)
        st.markdown("""
        **Random Forest Classifier** and **Gradient Boosting Classifier** shows **similar performance**, 
        whereas **logistic regression** has the poorest score across all metrics
        """)

    st.write("### ")
    st.divider()
    st.write('### Model and Seed Comparison Data')
    data = load_dataset('data/all_model_seed_scores.csv')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataframe = dataframe_explorer(data)
    st.dataframe(dataframe, use_container_width=True)


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Findings from Experiment")
    st.write('Please choose the section or objective you are interested in. 👇')
    st.write("### ")

    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Computation Time", description=""),
        stx.TabBarItemData(id=2, title="Solution Quality Assessment", description=""),
        stx.TabBarItemData(id=3, title="Exploratory Capability", description=""),
        stx.TabBarItemData(id=4, title="Model Performance", description="")
    ], default=1)

    if int(chosen_option) == 1:
        computation_time()
    elif int(chosen_option) == 2:
        quality_assessment()
    elif int(chosen_option) == 3:
        exploratory()
    elif int(chosen_option) == 4:
        model_performance()


load_page()