import streamlit as st
import streamlit.components.v1 as components
from st_pages import Page, show_pages


def load_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def load_html(file_name):
    with open(file_name):
        with open(file_name, 'r') as f:
            st.components.v1.html(f.read(), height=600)


def load_page_navigation():
    navigation = show_pages(
                    [
                        Page("../section/home.py", "Home", "🏠"),
                        Page("section/tuning_method_intro", "Tuning Methods"),
                    ]
                )
    return navigation

