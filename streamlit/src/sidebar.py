import streamlit as st
from numpy import zeros


def member(name, image_name, linkedin_url):
    with st.container(border=True):
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            try:
                st.image(f"images/{image_name}")
            except:
                st.image(zeros((200, 200, 3)))
        with col2:
            st.markdown(
                f"""<p style="margin-bottom:0px"><strong>{name}</strong></p>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
				<a href="{linkedin_url}" target="_blank">
					<img src="https://static-exp1.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" alt="LinkedIn Profile" width=25px> <span style="font-weight : bold">LinkedIn</span>
				</a>					
				""",
                unsafe_allow_html=True,
            )


def sidebar(auto_expand=False):
    with st.sidebar:
        st.markdown(
            """<h2 style="text-align: center">Réaliser par :</h2>""",
            unsafe_allow_html=True,
        )
        member(
            "Guillaume M'BALI",
            "guillaume.png",
            "https://www.linkedin.com/in/guillaumembali/",
        )
        st.divider()
        st.markdown(
            """<p style="text-align: center">&#128640; Version 1.3.1</p>""",
            unsafe_allow_html=True,
        )
