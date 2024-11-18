import streamlit as st
import os
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import glob
from scipy.io import loadmat
import plotly.graph_objects as go

def find_file_path(base_path, id, side):
    pattern = os.path.join(base_path, f"{id}_*_SAG_3D_DESS_{side}")
    matching_files = glob.glob(pattern)
    if matching_files:
        return matching_files[0]
    else:
        raise FileNotFoundError(f"No matching file found for pattern: {pattern}")

def create_custom_colormap():
    n_bins = 256
    colors = plt.cm.jet(np.linspace(0, 1, n_bins))[::-1]  # Reverse the colors
    colors[0] = [0.3, 0.3, 0.3, 1.0]  # Set the first color to grey for NaN values

    # Increase brightness by scaling RGB values
    for i in range(len(colors)):
        colors[i][:3] = np.clip(colors[i][:3] * 1.2, 0, 1)  # Increase brightness by 20%

    # Return a ListedColormap object
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

def load_and_process_mesh(stl_path, thickness_path):
    mesh = pv.read(stl_path)
    thickness_data = pd.read_csv(thickness_path, header=None, names=['thickness'])
    mesh.point_data['thickness'] = thickness_data['thickness'].values
    return mesh, thickness_data['thickness']

def plot_mesh_with_plotly(mesh, global_vmin, global_vmax, title):
    x, y, z = mesh.points.T
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    thickness = mesh.point_data['thickness']

    # Create a custom colorscale
    colormap = create_custom_colormap()
    colors = colormap.colors  # Extract the colors array from the colormap

    # Create the colorscale for Plotly
    colorscale = [[i / (len(colors) - 1), f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{a})'] 
                  for i, (r, g, b, a) in enumerate(colors)]
    
    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z,
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=thickness,
        colorscale=colorscale,
        cmin=global_vmin,
        cmax=global_vmax,
        colorbar_title='Thickness',
        showscale=False,
        flatshading=False,  # Enables smooth shading
        lighting=dict(
            ambient=0.6,
            diffuse=0.9,
            specular=0.2,
            roughness=0.8,
            fresnel=0.05
        ),
        lightposition=dict(
            x=0,
            y=0,
            z=100   # Adjust light position to illuminate the mesh better
        )
    )])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24)  # Increased font size for title
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',

        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def stl_viewer_page():
  

    if 'selected_id' not in st.session_state or st.session_state.selected_id is None:
        st.warning("Please select an ID on the ID Selection page first.")
        return

    selected_id = st.session_state.selected_id
    selected_side = st.session_state.selected_side

    # Time point selection
    time_points = ['00m', '12m', '24m', '36m', '48m', '72m', '96m']
    selected_time_point = st.select_slider("Select Time Point", time_points)

    # Calculate separate vmin and vmax for tibia and femur across all time points
    thickness_data = {'femur': [], 'tibia': []}
    thickness_maps = []  # Store all 2D maps

    for time_point in time_points:
        base_folder = os.path.join(st.session_state.data_path, 
                                   'DATA/processed_PP', 
                                   time_point)
        try:
            full_path = find_file_path(base_folder, selected_id, selected_side)
            # Load 2D thickness map for this time point
            thickness_maps.append(load_2d_map(full_path))

            for bone in ['femur', 'tibia']:
                _, thickness = load_and_process_mesh(
                    os.path.join(st.session_state.data_path, 'DATA', bone + '_ref_final.stl'),
                    os.path.join(full_path, os.path.basename(full_path) + f'_{bone}_cartThickness.txt')
                )
                thickness_data[bone].append(thickness)
        except FileNotFoundError:
            st.warning(f"No data found for ID {selected_id} at time point {time_point}")

    # Calculate global min and max across all data
    all_3d_thickness = np.concatenate([np.concatenate(thickness_data['femur']), 
                                       np.concatenate(thickness_data['tibia'])])
    all_2d_thickness = np.concatenate([map.flatten() for map in thickness_maps])
    global_vmin = min(np.nanmin(all_3d_thickness), np.nanmin(all_2d_thickness))
    global_vmax = max(np.nanmax(all_3d_thickness), np.nanmax(all_2d_thickness))

    # Create a layout with 3 columns
    col1, col2, col3 = st.columns(3)

    try:
        base_folder = os.path.join(st.session_state.data_path, 
                                   'DATA/processed_PP', 
                                   selected_time_point)
        full_path = find_file_path(base_folder, selected_id, selected_side)

        # Load 2D thickness map
        thickness_map = load_2d_map(full_path)

        # Femur visualization in first column
        with col1:
            st.subheader("Femur 3D reference")
            mesh, _ = load_and_process_mesh(
                os.path.join(st.session_state.data_path, 'DATA', 'femur_ref_final.stl'),
                os.path.join(full_path, os.path.basename(full_path) + '_femur_cartThickness.txt')
            )

            fig = plot_mesh_with_plotly(mesh, global_vmin, global_vmax, f"{selected_id} - {selected_time_point} - {selected_side}")
            st.plotly_chart(fig, use_container_width=True)

        # Tibia visualization in second column
        with col2:
            st.subheader("Tibia 3D reference")
            mesh, _ = load_and_process_mesh(
                os.path.join(st.session_state.data_path, 'DATA', 'tibia_ref_final.stl'),
                os.path.join(full_path, os.path.basename(full_path) + '_tibia_cartThickness.txt')
            )

            fig = plot_mesh_with_plotly(mesh, global_vmin, global_vmax, "")
            st.plotly_chart(fig, use_container_width=True)

        # 2D thickness map in third column
        with col3:
            st.subheader("2D Thickness Map")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='black')
            ax.set_facecolor('black')
            im = ax.imshow(thickness_map, cmap=create_custom_colormap(),
                           vmin=global_vmin, vmax=global_vmax)
            cbar = plt.colorbar(im)
            cbar.ax.set_facecolor('black')
            cbar.ax.tick_params(colors='white')
            ax.axis('off')
            st.pyplot(fig, use_container_width=False)
            plt.close()

    except FileNotFoundError:
        st.error(f"No data found for ID {selected_id} at time point {selected_time_point}")

if __name__ == "__main__":
    stl_viewer_page()
