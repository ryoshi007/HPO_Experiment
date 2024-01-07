import streamlit as st
from customtool import smalltools
import extra_streamlit_components as stx


def load_page():
    st.set_page_config(
        # page_title="Tuning Methods Intro",
        page_icon=smalltools.page_icon(),
        layout='wide'
    )

    smalltools.hide_unused_pages()
    st.title("Get Insight about Dataset")
    st.write('Please choose the section you are interested in. 👇')
    st.write("### ")

    chosen_option = stx.tab_bar(data=[
        stx.TabBarItemData(id=1, title="Dataset Description", description=""),
        stx.TabBarItemData(id=2, title="Exploratory Data Analysis", description=""),
    ], default=1)

load_page()