import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
from streamlit_embedcode import github_gist
from streamlit_extras.badges import badge


def introduction():
    st.markdown("""
    ## What is Grid Search?
    
    Grid Search is a technique used for hyperparameter tuning in machine learning models. It involves 
    systematically working through multiple combinations of parameter tunes, cross-validating as it goes to 
    determine which tune gives the best performance.
    """)


def process():
    st.markdown("""
    ## How Does Grid Search Work?
    
    1. **Define Parameter Grid**: Create a grid of hyperparameter values and specify the different values you want 
    to try for each hyperparameter.
    2. **Model Training**: Train a model for each combination of hyperparameters in the grid.
    3. **Cross-Validation**: Evaluate each model using cross-validation and record the performance.
    4. **Select Best Parameters**: Choose the hyperparameter combination that results in the best evaluation metric.
    """)


def show_parameter():
    st.markdown("""
    ## Specific Parameters in Grid Search
    
    No specific parameter for this algorithm.
    """)


def comparison():
    left_co, right_co = st.columns(2)
    with left_co:
        st.markdown("""
        ## Advantages
        
        - **Exhaustive**: Tests every single combination in the parameter space.
        - **Simple**: Easy to understand and implement.
        - **Parallelizable**: Each model training is independent and can be run in parallel.
        """)

    with right_co:
        st.markdown("""
        ## Disadvantages
        
        - **Computationally Intensive**: Can be very slow with large datasets or many parameters.
        - **Limited by Resolution**: Only as good as the granularity of the hyperparameter grid.
        """)


def practice():
    st.markdown("""
    ## Best Practices
    
    - **Pre-test Individual Parameters**: Narrow down the range of values for each hyperparameter.
    - **Use a Coarse Grid First**: Start with a coarse grid to identify the promising regions.
    - **Refine the Grid**: Once the best regions are identified, use a finer grid to hone in on the best 
    hyperparameters.
    """)


def github_code():
    badge(type="github", name="ryoshi007/Hyperparameter-Tuning")
    st.write('Adjust the width of code viewer')
    width = st.slider("Width", 300, 1420, 1420)
    github_gist('https://gist.github.com/ryoshi007/0e9ea0d36eeeafdc2d46c20793dff504', width=width)


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Grid Search")
    st.markdown("## ")

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image('static/grid_search_gif.gif')

    st.markdown("# ")
    st.markdown("### ")

    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Introduction", description=""),
        stx.TabBarItemData(id=2, title="Process", description=""),
        stx.TabBarItemData(id=3, title="Parameter", description=""),
        stx.TabBarItemData(id=4, title="Advantages & Disadvantages", description=""),
        stx.TabBarItemData(id=5, title="Ideal Practices", description=""),
        stx.TabBarItemData(id=6, title="GitHub Code", description="")
    ], default=1)

    if int(chosen_option) == 1:
        introduction()
    elif int(chosen_option) == 2:
        process()
    elif int(chosen_option) == 3:
        show_parameter()
    elif int(chosen_option) == 4:
        comparison()
    elif int(chosen_option) == 5:
        practice()
    elif int(chosen_option) == 6:
        github_code()


load_page()