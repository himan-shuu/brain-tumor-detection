import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tempfile
import os

# --- 1. Define Custom Loss & Metrics ---
# These functions must match the ones defined in your training notebook exactly.
# Without them, Keras will not know how to load the model architecture.

def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky_index

# --- 2. Model Loading Logic ---

@st.cache_resource
def load_segmentation_model():
    """
    Tries to load 'best_model_tversky.keras' first (best weights).
    Falls back to 'brain_tumor_segmentation_model.h5' (final epoch).
    """
    model_files = ['best_model_tversky.keras', 'brain_tumor_segmentation_model.h5']
    
    for file_name in model_files:
        if os.path.exists(file_name):
            try:
                # Load model with custom objects dictionary
                model = load_model(
                    file_name,
                    custom_objects={
                        'dice_coef': dice_coef,
                        'tversky_loss': tversky_loss
                    }
                )
                print(f"Successfully loaded: {file_name}")
                return model
            except Exception as e:
                st.error(f"Failed to load {file_name}: {e}")
    
    st.error(f"No model file found. Please ensure one of the following exists: {model_files}")
    return None

# --- 3. Preprocessing Functions ---

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to a temporary location for nibabel to read."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error processing file upload: {e}")
        return None

def preprocess_slice(slice_data):
    """
    Normalizes the input slice.
    Logic: (img - mean) / std per channel.
    Input shape: (240, 240, 4) -> Output shape: (1, 240, 240, 4)
    """
    processed = np.zeros_like(slice_data, dtype=np.float32)
    
    for i in range(4):
        img = slice_data[..., i]
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            processed[..., i] = (img - mean) / (std + 1e-8)
        else:
            processed[..., i] = img
            
    # Add batch dimension for the model
    return np.expand_dims(processed, axis=0)

# --- 4. Streamlit User Interface ---

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")

st.title("🧠 Brain Tumor Segmentation App")
st.markdown("""
**Model used:** U-Net with Tversky Loss.
**Instructions:** Upload the 4 separate MRI modality files for a single patient to generate a tumor segmentation mask.
""")

# Sidebar for file uploads
st.sidebar.header("1. Upload MRI Files (NIfTI)")
flair_file = st.sidebar.file_uploader("Upload FLAIR (.nii)", type=['nii', 'nii.gz'])
t1_file = st.sidebar.file_uploader("Upload T1 (.nii)", type=['nii', 'nii.gz'])
t1ce_file = st.sidebar.file_uploader("Upload T1ce (.nii)", type=['nii', 'nii.gz'])
t2_file = st.sidebar.file_uploader("Upload T2 (.nii)", type=['nii', 'nii.gz'])

# Main execution
if flair_file and t1_file and t1ce_file and t2_file:
    with st.spinner("Processing MRI Data..."):
        # Save uploads to temp files
        path_flair = save_uploaded_file(flair_file)
        path_t1 = save_uploaded_file(t1_file)
        path_t1ce = save_uploaded_file(t1ce_file)
        path_t2 = save_uploaded_file(t2_file)

        # Load volumes using Nibabel
        vol_flair = nib.load(path_flair).get_fdata()
        vol_t1 = nib.load(path_t1).get_fdata()
        vol_t1ce = nib.load(path_t1ce).get_fdata()
        vol_t2 = nib.load(path_t2).get_fdata()

        # Clean up temp files to save space
        for p in [path_flair, path_t1, path_t1ce, path_t2]:
            if os.path.exists(p):
                os.remove(p)

        # Stack into single volume (H, W, Depth, Channels)
        # Result shape: (240, 240, 155, 4)
        combined_volume = np.stack([vol_flair, vol_t1, vol_t1ce, vol_t2], axis=-1)
        max_slices = combined_volume.shape[2]

        st.success("✅ MRI Data Loaded Successfully!")

    # --- Slice Selection ---
    st.sidebar.header("2. View Slices")
    slice_index = st.sidebar.slider(
        "Select Slice Depth (Z-axis)", 
        min_value=0, 
        max_value=max_slices-1, 
        value=max_slices//2
    )

    # Extract 2D slice for the model
    # Shape: (240, 240, 4)
    current_slice = combined_volume[:, :, slice_index, :]

    # Run Prediction
    model = load_segmentation_model()
    
    if model:
        # Preprocess
        input_tensor = preprocess_slice(current_slice)
        
        # Predict
        pred_prob = model.predict(input_tensor, verbose=0)
        
        # Post-process (Thresholding)
        pred_mask = (pred_prob[0, :, :, 0] > 0.5).astype(np.uint8)
        
        # Calculate approximate tumor area in this slice
        tumor_pixels = np.sum(pred_mask)

        # --- Display Results ---
        st.markdown(f"### Analysis for Slice {slice_index}")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(current_slice[:, :, 0], caption="Input MRI (FLAIR Channel)", clamp=True, use_container_width=True)

        with col2:
            st.image(pred_mask * 255, caption="Generated Segmentation Mask", clamp=True, use_container_width=True)

        with col3:
            # Create overlay
            # Use matplotlib for better transparency/overlay control
            fig, ax = plt.subplots()
            ax.imshow(current_slice[:, :, 0], cmap='gray')
            # Overlay red mask where tumor is detected
            ax.imshow(pred_mask, cmap='Reds', alpha=0.5 if tumor_pixels > 0 else 0)
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
            st.caption("Overlay (MRI + Mask)")

        # Result Banner
        if tumor_pixels > 0:
            st.error(f"**Tumor Detected** in this slice. (Area: {tumor_pixels} pixels)")
        else:
            st.success("**No Tumor Detected** in this slice.")

else:
    st.info("👋 Waiting for all 4 MRI modalities to be uploaded in the sidebar.")