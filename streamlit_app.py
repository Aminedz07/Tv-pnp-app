import streamlit as st


# Title of the app
st.title("My Streamlit App")
st.write("Hello world ")
    
# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a page", ["Home", "About", "Contact"])

if page == "Home":
    st.subheader("Welcome to My App")
    st.write("This is the home page.")
elif page == "About":
    st.subheader("About")
    st.write("This app is built using Streamlit.")
elif page == "Contact":
    st.subheader("Contact")
    st.write("You can reach us at contact@example.com")
