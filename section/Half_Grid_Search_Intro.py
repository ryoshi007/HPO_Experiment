import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
from streamlit_embedcode import github_gist
from streamlit_extras.badges import badge


def introduction():
    st.markdown("""
    ## What is Successive Halving / Half-Grid Search?

    Successive Halving is a method used in hyperparameter tuning, especially in conjunction with grid search. This 
    technique is part of the broader family of algorithms known as bandit-based methods and is designed to be more 
    efficient than traditional grid search.
    """)


def process():
    st.markdown("""
    ## How Does Successive Halving / Half-Grid Search Work?

    1. **Start with a Broad Set**: Begin with a wide range of hyperparameter combinations.
    2. **Initial Resource Allocation**: Allocate a small budget (like a limited number of iterations or a subset of 
    the data) to each combination.
    3. **Performance Evaluation**: Assess the performance of each combination with the given budget.
    4. **Keep the Best, Halve the Rest**: Discard the lower-performing half of the combinations and double the 
    resource budget for the remaining candidates.
    5. **Iterate Until Convergence**: Repeat the halving process until a satisfactory combination is found or 
    resources are exhausted.
    """)


def comparison():
    left_co, right_co = st.columns(2)
    with left_co:
        st.markdown("""
        ## Advantages

        - **Resource Efficiency**: Focuses computational resources on the most promising hyperparameter combinations.
        - **Scalability**: More practical for large hyperparameter spaces as it quickly narrows down the search.
        - **Faster Convergence**: Can reach optimal or near-optimal solutions more quickly than exhaustive grid search.
        """)

    with right_co:
        st.markdown("""
        ## Disadvantages

        - **Potential to Miss Optimal Combinations**: Early rounds with limited resources might eliminate some 
        potentially optimal combinations.
        - **Dependence on Initial Budget**: The initial allocation of resources can influence which combinations are 
        retained.
        """)


def show_parameter():
    st.markdown("""
    ## Specific Parameters in Half Grid Search
    
    - **Factor**
        - Represent the reduction factor for culling parameter combinations.
        - Determine how aggressively the parameter grid is reduced in each iteration.
        - A higher factor value means more aggressive reduction, resulting in fewer parameter combinations being evaluated 
        in subsequent iterations.
        - Default value is set to 2, meaning that the number of parameter combinations is halved in each iteration.    
    """)
    st.markdown("### ")
    st.markdown("""
    - **Budget**:
        - Refer to the number of samples or data points used to evaluate each parameter combination.
        - Initially, a subset of the entire dataset (variable `iter_n_samples variable`), is used to evaluate the parameter combinations.
        - As iterations progress, the 'budget' increases, meaning more data points are used to evaluate each remaining parameter combination.
        - Help in quickly eliminating unpromising parameter combinations in the initial iterations with a smaller dataset, 
        and focusing computational resources on evaluating the most promising combinations with a larger dataset in later iterations.
    """)


def practice():
    st.markdown("""
    ## Best Practices

    - **Combine with Grid Search**: Use Successive Halving to narrow down the field in grid search.
    - **Dynamic Resource Allocation**: Adjust the resource budget based on the performance in each iteration.
    - **Monitor and Adjust**: Keep an eye on the early rounds to ensure promising combinations are not prematurely 
    discarded.
    """)


def github_code():
    badge(type="github", name="ryoshi007/Hyperparameter-Tuning")
    st.write('Adjust the width of code viewer')
    width = st.slider("Width", 300, 1420, 1420)
    github_gist('https://gist.github.com/ryoshi007/ca6f0f18ee567e8b3d58520b255aa7c3', width=width)


def load_page():
    st.set_page_config(
        # page_title="Tuning Methods Intro",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Half Grid Search")
    st.markdown("## ")

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image('static/half_grid_search_gif.gif')

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