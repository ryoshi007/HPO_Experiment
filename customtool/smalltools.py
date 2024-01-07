import time
import streamlit as st
import streamlit.components.v1 as components
from st_pages import hide_pages, show_pages_from_config
from PIL import Image


def load_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def load_html(file_name):
    with open(file_name):
        with open(file_name, 'r') as f:
            st.components.v1.html(f.read(), height=600)


def hide_unused_pages():
    show_pages_from_config()
    hide_pages(["Grid Search Intro", "Half Grid Search Intro", "Simulated Annealing Intro", "Genetic Algorithm Intro"])
    time.sleep(0.1)


def page_icon():
    img = Image.open('static/logo.png')
    return img