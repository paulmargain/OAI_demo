import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import glob
from matplotlib.colors import ListedColormap
import pandas as pd

def create_custom_colormap():
    n_bins = 256
    colors = plt.cm.jet(np.linspace(0, 1, n_bins))[::-1]  # Reverse the colors
    colors[0] = [0.5, 0.5, 0.5, 1.0]  # Set the first color to grey for NaN values
    return ListedColormap(colors)

def load_2d_map(full_path):
    """Load and process the 2D thickness map from .mat file"""
    mat_file = os.path.join(full_path, '2d_map_rois_CTh_auto.mat')
    data = loadmat(mat_file)
    
    # Combine femur and tibia maps
    femur_map = data['femur_2d_map_CTh']
    tibia_map = data['tibia_2d_map_CTh']
    combined_map = np.vstack([femur_map, tibia_map[0:50, 0:80]])
    return combined_map

def find_file_path(base_path, id, side):
    pattern = os.path.join(base_path, f"{id}_*_SAG_3D_DESS_{side}")
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]
    else:
        raise FileNotFoundError(f"No matching file found for pattern: {pattern}")

def comp_viewer_page():


    if st.session_state.selected_id is None:
        st.warning("Please select an ID on the ID Selection page first.")
        return

    selected_id = st.session_state.selected_id
    selected_side = st.session_state.selected_side

    # Time points to compare
    time_points = ['00m', '12m', '24m', '36m', '48m', '72m', '96m']
    time_values = [int(tp.replace('m', '')) for tp in time_points]

    # Store all thickness maps and calculate global min/max
    thickness_maps = []
    all_data = []

    for time_point in time_points:
        base_folder = os.path.join(st.session_state.data_path, 
                                   'DATA/processed_PP', 
                                   time_point)
        try:
            full_path = find_file_path(base_folder, selected_id, selected_side)
            thickness_map = load_2d_map(full_path)
            thickness_maps.append(thickness_map)
            all_data.extend(thickness_map.flatten())
        except FileNotFoundError:
            thickness_maps.append(None)

    # Calculate global min and max for consistent colormap
    valid_data = [x for x in all_data if not np.isnan(x)]
    if valid_data:
        global_vmin = np.min(valid_data)
        global_vmax = np.max(valid_data)
    else:
        st.error("No valid data found for the selected ID")
        return

    # Check if any maps are available
    n_plots = len([m for m in thickness_maps if m is not None])
    if n_plots == 0:
        st.error("No data available for the selected ID")
        return

    # Load the score and KL_grade data
    df = pd.read_csv(os.path.join(st.session_state.data_path, 'DATA/df_score_kl_temp2.28.csv'))
    selected_df = df[(df['ID'] == selected_id) & (df['SIDE'] == selected_side)]
    
    # Align the time points with the DataFrame
    scores = []
    kl_grades = []
    for time_point in time_points:
        row = selected_df[selected_df['Timepoint'] == time_point]
        if not row.empty:
            score = row.iloc[0]['score']
            kl_grade = row.iloc[0]['KL_grade']
        else:
            score = np.nan
            kl_grade = np.nan
        scores.append(score)
        kl_grades.append(kl_grade)

    # Create a figure with subplots for thickness maps
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(4 * n_plots, 8))  # Reduced height since we don't need the plot anymore
    gs = fig.add_gridspec(1, n_plots)  # Changed to single row

    # Plot the thickness maps
    axes_maps = []
    plot_idx = 0
    for i, (time_point, thickness_map) in enumerate(zip(time_points, thickness_maps)):
        if thickness_map is not None:
            ax = fig.add_subplot(gs[0, plot_idx])
            im = ax.imshow(thickness_map, cmap=create_custom_colormap(),
                           vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f"Time Point: {time_point}\nScore: {scores[i]:.1f}\nKL Grade: {kl_grades[i]:.0f}", 
                        color='white', pad=10, fontsize=20, y=-0.3)
            ax.axis('off')
            axes_maps.append(ax)
            plot_idx += 1

    # Adjust colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted position
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Thickness', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')

    # Adjust layout
    plt.subplots_adjust(wspace=0.1)

    # Show the figure
    st.pyplot(fig, use_container_width=False)
