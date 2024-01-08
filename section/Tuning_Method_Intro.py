import streamlit as st
from customtool import smalltools
from streamlit_card import card
from streamlit_extras.switch_page_button import switch_page
import base64


@st.cache_resource
def load_and_encode_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data)
    return "data:image/png;base64," + encoded.decode("utf-8")


def create_card(title, text, image_path, click_function):
    data = load_and_encode_image(image_path)

    card_component = card(
        title=title,
        text=text,
        image=data,
        on_click=click_function,
        styles={
            "card": {
                "overflow": "hidden",
                "position": "relative",
            },
            "image": {
                "object-fit": "cover",
                "width": "100%",
                "height": "100%",
                "position": "absolute"
            }
        }
    )
    return card_component


def load_page():
    st.set_page_config(
        page_title="HPO Experiment",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()

    st.markdown("# Tuning Methods Intro")
    st.markdown("###")
    st.markdown("### Please choose one hyperparameter tuning method below 👇")
    st.write('You can check out the codes on using these different hyperparameter tuning methods in Tuning Lab.')

    card_col1, card_col2, card_col3, card_col4 = st.columns(spec=4, gap='small')
    with card_col1:
        grid_search_card = create_card('Grid Search', '', 'static/grid_search_gif.gif', lambda: switch_page("Grid Search Intro"))

    with card_col2:
        half_grid_search_card = create_card('Half-Grid Search', '', 'static/half_grid_search_gif.gif', lambda: switch_page("Half Grid Search Intro"))

    with card_col3:
        simulated_annealing_card = create_card('Simulated Annealing', '', 'static/simulated_annealing_gif.gif', lambda: switch_page("Simulated Annealing Intro"))

    with card_col4:
        genetic_algo_card = create_card('Genetic Algorithm', '', 'static/genetic_algo_gif.gif', lambda: switch_page("Genetic Algorithm Intro"))


load_page()



