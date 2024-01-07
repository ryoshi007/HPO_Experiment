import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx
from streamlit_embedcode import github_gist
from streamlit_extras.badges import badge


def introduction():
    st.markdown("""
    ## What is Genetic Algorithm?

    A Genetic Algorithm (GA) is an optimization technique inspired by the process of natural selection. In machine 
    learning, it is used for hyperparameter tuning, where the goal is to find the best combination of 
    hyperparameters that optimizes a given objective function.
    """)


def process():
    st.markdown("""
    ## How Does Genetic Algorithm Work?

    1. **Initial Population**: Start with a randomly generated population of hyperparameter sets (called individuals).
    2. **Fitness Evaluation**: Each individual in the population is evaluated based on how well it performs the 
    given task (its fitness).
    3. **Selection**: Select the fittest individuals to be parents for generating the next population.
    4. **Crossover**: Combine pairs of parents to create offspring, which are new sets of hyperparameters.
    5. **Mutation**: Introduce random changes to some offspring to maintain genetic diversity.
    6. **New Generation**: The next generation of individuals (offspring) replaces the old one.
    7. **Repeat**: Steps 2 through 6 are repeated for several generations.
    """)


def show_parameter():
    st.markdown("""
    ## Specific Parameters in Genetic Algorithm

    - **Population Size**
        - Determine the number of parameter sets (or chromosomes) in each generation of the genetic algorithm.
        - A larger population size allows for greater diversity in the parameter sets, potentially leading to better 
        exploration of the parameter space, but at the cost of increasing computational complexity.
        - Remain constant across generations.
    """)
    st.markdown("### ")
    st.markdown("""
    - **Max Iterations**:
        - Specify the maximum number of generations the genetic algorithm will run.
        - Set an upper bound on the number of times the algorithm will iterate through the process of selection, 
        crossover, mutation, and generation creation.  
        - A higher number of iterations allows for more extensive exploration but also increases computational time.  
    """)
    st.markdown("### ")
    st.markdown("""
    - **No Improvement Limit**:
        - A stopping criterion based on the number of consecutive iterations without improvement in the best score.
        - If the best score does not improve for a specified number of consecutive iterations, the algorithm terminates 
        early, assuming it has reached a plateau.
        - Help in preventing unnecessary computations after the algorithm has likely converged.
    """)
    st.markdown("### ")
    st.markdown("""
    - **Selection Rate**:
        - Determine what proportion of the population is selected for breeding in each generation.
        - A higher selection rate means more chromosomes are selected as parents, which can increase diversity in the 
        offspring but may also retain suboptimal chromosomes.
        - Selection process is yypically based on the fitness of the chromosomes, with higher fitness having a better 
        chance of being selected.
    """)
    st.markdown("### ")
    st.markdown("""
    - **Mutation Rate**:
        - Control how frequently mutations occur in the genetic algorithm process.
        - A probability that a given gene (parameter) in a chromosome (parameter set) will be randomly altered.
        - Introduce randomness and helps in exploring new areas of the parameter space, potentially avoiding local 
        optima.
        - A very high mutation rate can lead to erratic search behavior and slow convergence.
    """)


def comparison():
    left_co, right_co = st.columns(2)
    with left_co:
        st.markdown("""
        ## Advantages

        - **Global Search Capability**: Capable of exploring a wide range of solutions, increasing the chance of 
        finding the global optimum.
        - **Robustness**: Performs well in complex and high-dimensional search spaces.
        - **Parallelizable**: Different individuals can be evaluated in parallel.
        """)

    with right_co:
        st.markdown("""
        ## Disadvantages

        - **Computationally Intensive**: Can require significant computational resources, especially for large 
        populations or many generations.
        - **Parameter Sensitivity**: The performance can be sensitive to the settings of various parameters (e.g., 
        mutation rate, crossover rate).
        - **No Guarantee of Optimal Solution**: Like many optimization algorithms, there's no guarantee of finding the 
        absolute best solution.
        """)


def practice():
    st.markdown("""
    ## Best Practices

    - **Careful Parameter Tuning**: The parameters of the GA (like mutation rate, crossover rate, population size) 
    should be carefully tuned to the problem at hand.
    - **Diversity Maintenance**: Implement strategies to maintain diversity in the population to avoid premature 
    convergence.
    - **Hybrid Approaches**: Consider combining GAs with other optimization techniques to exploit their respective 
    strengths.
    """)


def github_code():
    badge(type="github", name="ryoshi007/Hyperparameter-Tuning")
    st.write('Adjust the width of code viewer')
    width = st.slider("Width", 300, 1420, 1420)
    github_gist('https://gist.github.com/ryoshi007/c6e58147b7b64b7f89d8aeee1d2cfb9e', width=width)


def load_page():
    st.set_page_config(
        # page_title="Tuning Methods Intro",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Genetic Algorithm")
    st.markdown("## ")

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image('static/genetic_algo_gif.gif')

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