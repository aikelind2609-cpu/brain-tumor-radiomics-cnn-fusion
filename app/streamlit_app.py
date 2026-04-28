"""
Brain Tumor MRI Classifier - Streamlit Demo
Hybrid Radiomics + Deep CNN Feature Fusion
"""

import streamlit as st
import joblib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label as sk_label
import mahotas
import io
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS = PROJECT_ROOT / 'models'
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_COLORS = {'glioma': '#1f77b4', 'meningioma': '#ff7f0e', 
                'notumor': '#2ca02c', 'pituitary': '#d62728'}

# ============================================================
# LOAD MODELS (cached so it loads once)
# ============================================================
@st.cache_resource
def load_models():
    rad_bundle = joblib.load(MODELS / 'radiomics.pkl')
    deep_bundle = joblib.load(MODELS / 'deep.pkl')
    fused_bundle = joblib.load(MODELS / 'fused.pkl')
    
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    resnet.fc = nn.Identity()
    resnet.eval()
    for p in resnet.parameters():
        p.requires_grad = False
    
    deep_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return rad_bundle, deep_bundle, fused_bundle, resnet, deep_preprocess

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================
def crop_brain(img_gray):
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img_gray
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    pad = 5
    h_img, w_img = img_gray.shape
    return img_gray[max(0, y-pad):min(h_img, y+h+pad), 
                    max(0, x-pad):min(w_img, x+w+pad)]

def preprocess_image(img_gray, size=224):
    img = crop_brain(img_gray)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def make_brain_mask(img_gray):
    _, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_big = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1: return np.ones_like(img_gray, dtype=np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = (labels == largest).astype(np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    brain_mask_filled = brain_mask.copy()
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(brain_mask_filled, contours, -1, 1, thickness=cv2.FILLED)
    return brain_mask_filled

def extract_first_order(img, mask):
    pixels = img[mask > 0].astype(np.float64)
    if len(pixels) == 0:
        return {f'fo_{k}': 0.0 for k in ['mean','std','min','max','median','p10','p90',
                                          'skew','kurt','energy','entropy','range','iqr']}
    p10, p25, p50, p75, p90 = np.percentile(pixels, [10,25,50,75,90])
    mean = pixels.mean(); std = pixels.std() + 1e-9
    skew = ((pixels-mean)**3).mean() / (std**3)
    kurt = ((pixels-mean)**4).mean() / (std**4) - 3
    hist, _ = np.histogram(pixels, bins=64, range=(0,256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return {'fo_mean':float(mean),'fo_std':float(pixels.std()),'fo_min':float(pixels.min()),
            'fo_max':float(pixels.max()),'fo_median':float(p50),'fo_p10':float(p10),
            'fo_p90':float(p90),'fo_skew':float(skew),'fo_kurt':float(kurt),
            'fo_energy':float((pixels**2).sum()/1e6),'fo_entropy':float(entropy),
            'fo_range':float(pixels.max()-pixels.min()),'fo_iqr':float(p75-p25)}

def extract_glcm(img, mask, levels=32):
    img_q = (img.astype(np.float32) * (levels-1) / 255).astype(np.uint8)
    img_q = np.where(mask > 0, img_q, 0)
    glcm = graycomatrix(img_q, distances=[1,3], angles=[0,np.pi/4,np.pi/2,3*np.pi/4],
                         levels=levels, symmetric=True, normed=True)
    feats = {}
    for prop in ['contrast','homogeneity','energy','correlation','dissimilarity','ASM']:
        vals = graycoprops(glcm, prop)
        feats[f'glcm_{prop}_mean'] = float(vals.mean())
        feats[f'glcm_{prop}_std']  = float(vals.std())
    return feats

def extract_haralick(img, mask):
    img_masked = np.where(mask > 0, img, 0).astype(np.uint8)
    try:
        h = mahotas.features.haralick(img_masked, ignore_zeros=True, return_mean=True)
    except Exception:
        h = np.zeros(13)
    names = ['angular_2nd_moment','contrast','correlation','sum_of_squares','inverse_diff_moment',
            'sum_avg','sum_variance','sum_entropy','entropy','difference_variance',
            'difference_entropy','info_correlation_1','info_correlation_2']
    return {f'har_{n}': float(v) for n, v in zip(names, h)}

def extract_shape(mask):
    mask_bin = (mask > 0).astype(np.uint8)
    labels = sk_label(mask_bin)
    if labels.max() == 0:
        return {f'sh_{k}': 0.0 for k in ['area','perimeter','eccentricity','solidity',
                                          'extent','major_axis','minor_axis','circularity','aspect_ratio']}
    region = max(regionprops(labels), key=lambda r: r.area)
    perim = region.perimeter + 1e-9
    circularity = 4 * np.pi * region.area / (perim**2)
    aspect = region.major_axis_length / (region.minor_axis_length + 1e-9)
    return {'sh_area':float(region.area),'sh_perimeter':float(region.perimeter),
            'sh_eccentricity':float(region.eccentricity),'sh_solidity':float(region.solidity),
            'sh_extent':float(region.extent),'sh_major_axis':float(region.major_axis_length),
            'sh_minor_axis':float(region.minor_axis_length),'sh_circularity':float(circularity),
            'sh_aspect_ratio':float(aspect)}

def extract_radiomics(img, mask):
    feats = {}
    feats.update(extract_first_order(img, mask))
    feats.update(extract_glcm(img, mask))
    feats.update(extract_haralick(img, mask))
    feats.update(extract_shape(mask))
    return feats

def extract_deep(img, resnet, deep_preprocess):
    pil_img = Image.fromarray(img).convert('L')
    tensor = deep_preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        feats = resnet(tensor).cpu().numpy().flatten()
    return feats

# ============================================================
# SIMILARITY SEARCH (Novelty: case-based retrieval)
# ============================================================
@st.cache_resource
def load_similarity_index():
    """Load the precomputed training set feature matrix."""
    return joblib.load(MODELS / 'similarity_index.pkl')

def find_similar_cases(query_features, top_k=3):
    """
    Find the top-k most similar training images using cosine similarity.
    
    query_features: (1, 300) array — selected features for the input image
    Returns: list of dicts with image_id, class, similarity score
    """
    index = load_similarity_index()
    train_features = index['features']  # (5600, 300)
    image_ids = index['image_ids']
    classes = index['classes']
    
    # Cosine similarity: dot product of normalized vectors
    query_norm = query_features / (np.linalg.norm(query_features) + 1e-9)
    train_norm = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-9)
    similarities = (train_norm @ query_norm.T).flatten()
    
    # Top k indices
    top_idx = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i in top_idx:
        results.append({
            'image_id': image_ids[i],
            'class': classes[i],
            'similarity': float(similarities[i]),
            'index': int(i),
        })
    return results

def find_image_path(image_id, class_name):
    """Locate the original training image on disk."""
    candidates = list((PROJECT_ROOT / 'data' / 'raw' / 'Training' / class_name).glob(f'{image_id}.*'))
    if candidates:
        return candidates[0]
    candidates = list((PROJECT_ROOT / 'data' / 'processed' / 'Training' / class_name).glob(f'{image_id}.*'))
    if candidates:
        return candidates[0]
    return None

# ============================================================
# UI
# ============================================================
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("**Hybrid Radiomics + Deep CNN Feature Fusion**")
st.markdown("Upload an MRI scan and get predictions from three different models.")

# Sidebar with project info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This demo runs a brain tumor classification pipeline that fuses:
    - **Hand-crafted radiomics features** (47 features: intensity, GLCM, Haralick, shape)
    - **Deep CNN features** (2048-d from frozen ResNet50 ImageNet weights)
    
    **Three classifiers compared:**
    1. Radiomics-only (XGBoost)
    2. Deep-only (XGBoost)
    3. Fused with SelectKBest(k=300) + XGBoost
    
    **Test set accuracy: ~89.94%**
    
    **Dataset:** Nickparvar Brain Tumor MRI (7,200 images, 4 classes)
    """)
    st.markdown("---")
    st.markdown("**Author:** Aikel Indurkhya")
    st.markdown("[GitHub repo](https://github.com/aikelind2609-cpu/brain-tumor-radiomics-cnn-fusion)")

# Load models with a spinner
with st.spinner("Loading models..."):
    rad_bundle, deep_bundle, fused_bundle, resnet, deep_preprocess = load_models()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a brain MRI scan in grayscale. The system will automatically preprocess it."
)

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file).convert('L')
    img_array = np.array(image)
    
    # Show original
    st.markdown("### Original Image")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=300)
    
    # Process
    with st.spinner("Preprocessing image and extracting features..."):
        proc_img = preprocess_image(img_array)
        mask = make_brain_mask(proc_img)
        rad_feats = extract_radiomics(proc_img, mask)
        deep_feats_arr = extract_deep(proc_img, resnet, deep_preprocess)
    
    with col_b:
        # Visualize preprocessing
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(proc_img, cmap='gray')
        axes[0].set_title("Preprocessed (224×224, CLAHE)")
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Brain Mask")
        axes[1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Build feature vectors
    rad_vec = np.array([rad_feats[c] for c in rad_bundle['cols']]).reshape(1, -1)
    deep_vec = deep_feats_arr.reshape(1, -1)
    fused_vec = np.concatenate([rad_vec, deep_vec], axis=1)
    
    # Apply scalers and selectors
    rad_scaled = rad_bundle['scaler'].transform(rad_vec)
    deep_scaled = deep_bundle['scaler'].transform(deep_vec)
    fused_scaled = fused_bundle['scaler'].transform(fused_vec)
    fused_selected = fused_bundle['selector'].transform(fused_scaled)
    
    # Predict
    with st.spinner("Running 3 classifiers..."):
        rad_proba = rad_bundle['clf'].predict_proba(rad_scaled)[0]
        deep_proba = deep_bundle['clf'].predict_proba(deep_scaled)[0]
        fused_proba = fused_bundle['clf'].predict_proba(fused_selected)[0]
    
    # Headline result
    fused_pred_idx = np.argmax(fused_proba)
    fused_pred = CLASS_NAMES[fused_pred_idx]
    fused_conf = fused_proba[fused_pred_idx]
    
    st.markdown("---")
    st.markdown("### 🎯 Final Prediction (Fused Model)")
    
    if fused_conf >= 0.85:
        confidence_level = "🟢 High confidence"
    elif fused_conf >= 0.65:
        confidence_level = "🟡 Medium confidence"
    else:
        confidence_level = "🔴 Low confidence (radiologist review recommended)"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", fused_pred.upper())
    col2.metric("Confidence", f"{fused_conf:.1%}")
    col3.metric("Status", confidence_level)
    
    # Side-by-side bar charts
    st.markdown("### 📊 Three-Model Comparison")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = [CLASS_COLORS[c] for c in CLASS_NAMES]
    
    for idx, (probs, name) in enumerate([
        (rad_proba, 'Radiomics only'),
        (deep_proba, 'Deep only'),
        (fused_proba, 'Fused (final)'),
    ]):
        pred_idx = np.argmax(probs)
        pred = CLASS_NAMES[pred_idx]
        conf = probs[pred_idx]
        axes[idx].bar(CLASS_NAMES, probs, color=colors)
        axes[idx].set_title(f"{name}\nPredicted: {pred} ({conf:.1%})")
        axes[idx].set_ylim(0, 1)
        axes[idx].set_ylabel('Probability')
        axes[idx].tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Detailed probabilities table
    with st.expander("📋 Detailed probabilities (all models)"):
        import pandas as pd
        prob_df = pd.DataFrame({
            'Class': CLASS_NAMES,
            'Radiomics-only': [f"{p:.4f}" for p in rad_proba],
            'Deep-only':       [f"{p:.4f}" for p in deep_proba],
            'Fused':           [f"{p:.4f}" for p in fused_proba],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ========== NOVELTY: Similar Cases ==========
    st.markdown("### 🔬 Similar Training Cases (Case-Based Retrieval)")
    st.caption("The 3 training images most similar to your input, in 300-d fused feature space.")
    
    with st.spinner("Searching 5,600 training images for similar cases..."):
        similar_cases = find_similar_cases(fused_selected, top_k=3)
    
    # Check whether all 3 agree with the model's prediction
    all_match = all(case['class'] == fused_pred for case in similar_cases)
    if all_match:
        st.success(f"✅ All 3 most similar training cases are also **{fused_pred}** — the prediction is well-supported by training data.")
    else:
        retrieved_classes = [case['class'] for case in similar_cases]
        if fused_pred in retrieved_classes:
            count = retrieved_classes.count(fused_pred)
            st.warning(f"⚠️ Only {count}/3 similar training cases match the predicted class **{fused_pred}**. Other cases: {retrieved_classes}")
        else:
            st.error(f"🚨 None of the 3 most similar training cases match **{fused_pred}**. Retrieved classes: {retrieved_classes}. The prediction may be unreliable.")
    
    # Display similar images side-by-side
    sim_cols = st.columns(3)
    for i, case in enumerate(similar_cases):
        with sim_cols[i]:
            img_path = find_image_path(case['image_id'], case['class'])
            if img_path is not None:
                sim_img = Image.open(img_path).convert('L')
                st.image(sim_img, use_container_width=True)
                
                # Color-code by class match
                if case['class'] == fused_pred:
                    st.markdown(f"**Class:** `{case['class']}` ✅")
                else:
                    st.markdown(f"**Class:** `{case['class']}` ⚠️")
                st.markdown(f"**Similarity:** {case['similarity']:.3f}")
                st.caption(f"ID: {case['image_id']}")
            else:
                st.warning(f"Image not found on disk: {case['image_id']}")
                st.markdown(f"**Class:** `{case['class']}`")
                st.markdown(f"**Similarity:** {case['similarity']:.3f}")
    
    st.markdown("---")

    # Top features used
    with st.expander("🔍 Top features for this prediction"):
        # Get the actual feature values (selected)
        feat_values = fused_selected[0]
        feat_names = fused_bundle['selected_names']
        
        # Multiply by feature importance to estimate contribution
        importances = fused_bundle['clf'].feature_importances_
        contributions = np.abs(feat_values * importances)
        top_idx = np.argsort(contributions)[::-1][:10]
        
        contrib_df = pd.DataFrame({
            'Feature': [feat_names[i] for i in top_idx],
            'Type': ['radiomics' if not feat_names[i].startswith('deep_') else 'deep' 
                     for i in top_idx],
            'Importance × |Value|': [f"{contributions[i]:.4f}" for i in top_idx],
        })
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        st.caption("These features had the strongest combined importance × scaled-value for this image.")

else:
    st.info("👆 Upload a brain MRI image to get started.")
    st.markdown("""
    **Tips for best results:**
    - Use grayscale brain MRI scans (T1 or T2 weighted)
    - PNG, JPG, or JPEG formats supported
    - The system auto-crops, resizes to 224×224, and applies CLAHE contrast enhancement
    """)

st.markdown("---")
st.caption("⚠️ This is a research prototype, not a medical device. Not for clinical use.")