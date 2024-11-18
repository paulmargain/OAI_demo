import streamlit as st
import pandas as pd
import os
from PIL import Image

def id_selection_page():
    # Welcome message and page explanations
    st.title('Welcome to 3D Cartilage Thickness Viewer')
    
    st.markdown("""
    ### Available Pages:
    1. **ID Selection** (Current Page): Select between two sample cases
    2. **Image Viewer**: View DESS MRI scans with segmentation overlays across different time points
    3. **STL Viewer**: Visualize 3D cartilage thickness maps on reference bones
    4. **Maps Comparison**: Compare 2D thickness maps across all available time points
    
    
   
  
    ### Sample Cases:
    - Case 1: ID 900099 (LEFT knee)
    - Case 2: ID 9995338 (RIGHT knee)
    """)
    st.markdown("For any questions, you can email [paul.margain@chuv.ch](mailto:paul.margain@chuv.ch)")
    st.markdown("---")
    
    # Create two columns: one for selection, one for confirmation
    col_select, col_confirm = st.columns([0.6, 0.4])

    with col_select:
        # Simplified ID and side selection
        selected_case = st.radio(
            "Select Case",
            options=["Case 1: ID 900099 (LEFT)", "Case 2: ID 9995338 (RIGHT)"]
        )
        
        # Set ID and side based on selection
        if selected_case == "Case 1: ID 900099 (LEFT)":
            selected_id = 9000099
            selected_side = "LEFT"
        else:
            selected_id = 9995338
            selected_side = "RIGHT"

    with col_confirm:
        if st.button('Confirm Selection'):
            st.session_state.selected_id = selected_id
            st.session_state.selected_side = selected_side
            st.success(f"""
            Selected ID {selected_id} ({selected_side} knee)
            
            You can now explore:
            - Image Viewer: MRI scans with segmentations
            - STL Viewer: 3D thickness visualization
            - Maps Comparison: Thickness changes over time
            """)

# Run the app
if __name__ == '__main__':
    id_selection_page()
