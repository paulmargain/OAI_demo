import streamlit as st
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import nibabel as nib
import base64
from io import BytesIO
import glob

def load_nifti_file(filepath):
    nifti_img = nib.load(filepath)
    image_np = np.asanyarray(nifti_img.dataobj)
    spacing = nifti_img.header.get_zooms()[:3]
    return image_np, spacing

def normalize_mri(image):
    p1, p99 = np.percentile(image, (1, 99))
    image_normalized = np.clip(image, p1, p99)
    image_normalized = (image_normalized - p1) / (p99 - p1)
    return image_normalized

def apply_window(image, window_center, window_width):
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    return np.clip((image - min_val) / (max_val - min_val), 0, 1)

def create_custom_colormap():
    colors = [(0, 0, 0, 0),  # Transparent for background
              (1, 0, 0, 0.5),  # Red with 50% opacity for bone
              (0, 1, 0, 0.5),  # Green with 50% opacity for cartilage
              (1, 0, 0, 0.5),  # Red with 50% opacity for bone
              (0, 1, 0, 0.5)]  # Green with 50% opacity for cartilage
    return mcolors.ListedColormap(colors)

def plot_slice(slice, mask_slice, spacing=None, window_center=0.5, window_width=1.0, show_mask=True):
    aspect_ratio = spacing[1] / spacing[0] if spacing else 1
    fig, ax = plt.subplots(figsize=(6, 6))

    slice = np.rot90(slice)
    slice = apply_window(slice, window_center, window_width)
    
    ax.imshow(slice, cmap='gray', aspect=aspect_ratio)
    
    if show_mask:
        mask_slice = np.rot90(mask_slice)
        custom_cmap = create_custom_colormap()
        ax.imshow(mask_slice, cmap=custom_cmap, vmin=0, vmax=4, aspect=aspect_ratio)
    
    ax.axis('off')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_viewer_page():
    st.title("DESS Segmentation")

    if st.session_state.selected_id is None:
        st.warning("Please select an ID on the ID Selection page first.")
        return

    selected_id = st.session_state.selected_id

    # Time point selection
    time_points = ['00m', '12m', '24m', '36m', '48m', '72m','96m']
    selected_time_points = st.multiselect("Select Time Points (select up to 3)", time_points, default=['00m', '12m', '24m'])
    if len(selected_time_points) == 0:
        st.warning("Please select at least one time point.")
        return
    elif len(selected_time_points) > 3:
        st.warning("Please select up to 3 time points.")
        return

    # View selection
    st.sidebar.header("View Selection")
    view = st.sidebar.radio("Choose view", ["Sagittal", "Coronal", "Axial"])

    # Map view to dimension index
    view_to_dim = {'Axial': 2, 'Coronal': 1, 'Sagittal': 0}
    dim_idx = view_to_dim[view]

    images = []
    masks = []
    spacings = []
    shapes = []

    for tp in selected_time_points:
        # Build the image and mask paths
        image_pattern = os.path.join(st.session_state.data_path, 'IMAGE', tp, f"DESS_{tp}", f"{selected_id}_*_SAG_3D_DESS_*_0000.nii.gz")
       
        image_files = glob.glob(image_pattern)
        mask_pattern = os.path.join(st.session_state.data_path, 'DATA', 'pred', f"pred_{tp.lower()}_PP", f"{selected_id}_*_SAG_3D_DESS_*.nii.gz")
        mask_files = glob.glob(mask_pattern)
        
        if image_files and mask_files:
            image_path = image_files[0]
            mask_path = mask_files[0]
            # Load image and mask
            image_np, spacing = load_nifti_file(image_path)
            image_np = normalize_mri(image_np)
            mask_np, _ = load_nifti_file(mask_path)
            images.append(image_np)
            masks.append(mask_np)
            spacings.append(spacing)
            shapes.append(image_np.shape)
        else:
            st.error(f"No image or mask found for ID {selected_id} at time point {tp}")
            return

    # Get the minimum number of slices among the images in the selected dimension
    slices_per_image = [shape[dim_idx] for shape in shapes]
    min_slices = min(slices_per_image)

    slice_num = st.sidebar.slider(f"{view} Slice", 0, min_slices - 1, min_slices // 2)

    # Calculate default window_center and window_width based on the first image
    default_center = float(np.mean(images[0]))
    default_width = float(np.std(images[0]) * 2)  # Using 2 standard deviations

    # Add contrast adjustment controls with optimal defaults
    st.sidebar.header("Image Adjustment")
    window_center = st.sidebar.slider("Window Center", 0.0, 1.0, default_center, 0.01)
    window_width = st.sidebar.slider("Window Width", 0.01, 1.0, default_width, 0.01)
    
    # Add mask toggle
    show_mask = st.sidebar.checkbox("Show Mask", value=True)

    # Create columns
    cols = st.columns(len(selected_time_points))

    for i, col in enumerate(cols):
        image_np = images[i]
        mask_np = masks[i]
        spacing = spacings[i]
        # Get the slice
        if view == "Axial":
            slice_data = image_np[:, :, slice_num]
            mask_slice = mask_np[:, :, slice_num]
            spacing_2d = (spacing[0], spacing[1])
        elif view == "Coronal":
            slice_data = image_np[:, slice_num, :]
            mask_slice = mask_np[:, slice_num, :]
            spacing_2d = (spacing[0], spacing[2])
        else:  # Sagittal
            slice_data = image_np[slice_num, :, :]
            mask_slice = mask_np[slice_num, :, :]
            spacing_2d = (spacing[1], spacing[2])

        img_str = plot_slice(slice_data, mask_slice, spacing_2d, window_center=window_center, window_width=window_width, show_mask=show_mask)
        with col:
            col.markdown(f"""
            <div style="display: flex; justify-content: center; align-items: center;">
                <img id="mri-image" src="data:image/png;base64,{img_str}" style="max-width: 100%; object-fit: contain;">
            </div>
            """, unsafe_allow_html=True)
            col.write(f"Time Point: {selected_time_points[i]}")
