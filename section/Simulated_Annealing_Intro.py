import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
from streamlit_embedcode import github_gist
from streamlit_extras.badges import badge


def introduction():
    st.markdown("""
    ## What is Simulated Annealing?

    Simulated Annealing is an optimization technique inspired by the process of annealing in metallurgy. This 
    method is used for hyperparameter tuning in machine learning models, particularly when searching for a global 
    optimum in a large space of parameters.
    """)


def process():
    st.markdown("""
    ## How Does Simulated Annealing Work?

    1. **Start with an Initial Solution**: Begin with a random hyperparameter configuration.
    2. **Iterative Exploration**: At each iteration, a new configuration is randomly selected.
    3. **Acceptance Probability**: The new configuration is accepted based on a probability that depends on the 
    current temperature and the change in the cost function.
    4. **Cooling Schedule**: Gradually lower the 'temperature' over time, reducing the probability of accepting 
    worse solutions.
    5. **Convergence**: Continue the process until the system 'freezes' at a minimum energy state, i.e., no 
    further improvement is found.
    """)


def show_parameter():
    st.markdown("""
    ## Specific Parameters in Simulated Annealing

    - **Maximum Iterations**
        - Represent the maximum number of iterations the Simulated Annealing algorithm will perform.
        - Set a limit on how many different sets of parameters the algorithm will evaluate.
        - A higher max_iter allows the algorithm more opportunities to explore the parameter space, which could lead to 
        finding a better set of parameters but at the cost of increased computational time.
        - Default value is set to 100, which means the algorithm will perform a maximum of 100 iterations unless it 
        converges to an optimal solution earlier.   
    """)
    st.markdown("### ")
    st.markdown("""
    - **Initial Temperature**:
        - Starting temperature for the Simulated Annealing process.
        - A metaphorical concept that controls the probability of accepting worse solutions as the algorithm runs. 
        - A higher initial temperature allows the algorithm to explore the parameter space more freely, potentially avoiding local minima in the early stages.
        - As the temperature is high initially, the algorithm is more likely to accept both better and worse solutions. 
        This helps in a more extensive search but decreases as the temperature cools down.
        - Default value is 100, providing the algorithm with a relatively high level of initial "exploration" freedom.
    """)
    st.markdown("### ")
    st.markdown("""
    - **Cooling Rate**:
        - Specify how quickly the temperature decreases in each iteration.
        - It is a fraction between 0 and 1. 
        - A lower cooling rate means that the temperature decreases more slowly, allowing more exploration of the 
        parameter space, but also meaning that more iterations may be required for convergence.
        - A higher cooling rate cools the system faster, leading to quicker convergence but with a higher risk of getting stuck in local optima.
        - Default value is 0.95, indicating that the temperature decreases by 5% after each iteration.
    """)


def comparison():
    left_co, right_co = st.columns(2)
    with left_co:
        st.markdown("""
        ## Advantages

        - **Avoids Local Minima**: The probabilistic nature of simulated annealing helps to escape local minima.
        - **Flexibility**: Can be applied to a wide range of optimization problems.
        - **Simple to Implement**: The algorithm is straightforward and easy to understand.
        """)

    with right_co:
        st.markdown("""
        ## Disadvantages

        - **Computationally Intensive**: Can be slower than other methods due to the iterative process.
        - **Dependent on Cooling Schedule**: The efficiency of the algorithm heavily relies on the cooling schedule and 
        parameters chosen.
        - **No Guaranteed Optimal Solution**: The final solution may not always be the global optimum.
        """)


def practice():
    st.markdown("""
    ## Best Practices

    - **Careful Design of the Cooling Schedule**: The cooling schedule should be designed carefully to balance 
    exploration and exploitation.
    - **Tuning of Parameters**: Parameters such as the initial temperature and the rate of cooling need to be tuned 
    to the specific problem.
    - **Combination with Other Techniques**: Sometimes combined with other techniques for hyperparameter tuning to 
    enhance performance.
    """)


def github_code():
    badge(type="github", name="ryoshi007/Hyperparameter-Tuning")
    st.write('Adjust the width of code viewer')
    width = st.slider("Width", 300, 1420, 1420)
    github_gist('https://gist.github.com/ryoshi007/3ecfa4208051d9903fc0295a8b1ff1a8', width=width)


def load_page():
    st.set_page_config(
        # page_title="Tuning Methods Intro",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Simulated Annealing")
    st.markdown("## ")

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image('static/simulated_annealing_gif.gif')

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