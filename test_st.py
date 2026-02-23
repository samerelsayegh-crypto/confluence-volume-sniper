import streamlit as st
import time

if st.button("Run"):
    with st.spinner("spinning"):
        st.write("Inside spinner")
        tabs = st.tabs(["A", "B"])
        with tabs[0]:
            st.write("Tab A")
