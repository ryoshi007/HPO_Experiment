import streamlit as st
from customtool import smalltools


def show_sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Tuning Methods Intro"])
    return page


def main():
    st.set_page_config(
        # page_title='Main Page',
        page_icon=smalltools.page_icon(),
        layout='wide'
    )
    smalltools.hide_unused_pages()
    smalltools.load_css('styles/style.css')

    st.image('static/home_banner.png')
    st.divider()
    st.write("# Data Science Project")
    st.write("### ")
    st.write("This is a project collaborated with EY that aims to conduct comparative analysis between traditional hyperparameter tuning method - Grid Search "
             "with other novel optimization techniques, including half grid search (successive halving), simulated annealing, and genetic algorithm. "
             "The entire workflow are integrated deeply with CRISP-DM methodology when conducting the experiment.")
    st.write("#### ")
    st.write("Most of the hyperparameter tuning researches do not directly compare all of the mentioned techniques. Most of them focused one popular "
             "techniques like simulated annealing with genetic algorithm, naive bayes optimization with grid search and so on. There is less amount of paper "
             "that compare half grid search with these methods. Thus, this experiment is conducted to fill in the gap in research of hyperparameter tuning.")
    st.write("#### ")
    st.write("The objectives are as follows:")
    st.markdown("""
    1. **Measure** and **compare** average computation time for finding optimal set of hyperparameters
    2. **Assess** quality of hyperparameters found by each method with accuracy, precision, recall, F1-score and ROC AUC
    3. **Quantity** diversity of hyperparameters evaluated by each method to assess their exploratory capabilities.
    """)
    st.write("#### ")
    st.write("This Streamlit comprises of several results, findings, and products that came from the experiment. It includes several sections for")
    st.markdown("""
    1. **Dataset Overview and EDA** - Know about the dataset used in the experiment and EDAs related to it
    2. **HyperTune 101** - Learn about the hyperparameter tuning methods used in this experiment in details
    3. **Predict Churn** - Conduct credit card churn prediction with 4 best models produced from the experiment
    4. **Tuning Lab** - Produce your own hyperparameter tuning codes with guidance
    5. **Results and Findings** - Explore the results from the experiments along with visualization
    """)


if __name__ == '__main__':
    main()