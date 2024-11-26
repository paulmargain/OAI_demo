import streamlit as st
from streamlit_option_menu import option_menu
from page1_id_selection import id_selection_page
from page2_image_viewer import image_viewer_page
from page3_stl_viewer import stl_viewer_page
from page4_maps_comparaison import comp_viewer_page
import os

st.set_page_config(page_title='OAI Data Viewer', layout="wide")


# Add the custom CSS here

def main():

    st.session_state.data_path = './DATA/'
    
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    # Initialize session state for selected side
    if "selected_side" not in st.session_state:
        st.session_state.selected_side = 'LEFT'



    # Main navigation menu
    selected = option_menu(
        None, 
        ["ID Selection", "DESS Segmentation", "Model Viewer", "Comparaison of maps"],
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal"
    )

    if selected == "ID Selection":
        id_selection_page()
    elif selected == "DESS Segmentation":
        image_viewer_page()
    elif selected == 'Model Viewer':
        stl_viewer_page()
    else:
        comp_viewer_page()

if __name__ == "__main__":
    main()
  
