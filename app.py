import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel
import math
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(layout="wide", page_title="Image Scene Localization with CLIP")

st.title("Image Scene Localization")
st.markdown("Upload an image, enter a text query, and find the most relevant scene within the image.")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache the model and processor to avoid reloading on every rerun
@st.cache_resource
def load_clip_model():
    """Loads the CLIP model and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

model, processor, device = load_clip_model()

def generate_patch_sizes_aspect_ratios(img_width, img_height, base_unit=32, num_scales=6,
                                       min_fraction=0.1, max_fraction=0.6,
                                       aspect_ratios=[
                                           # tall rectangle (1:2)
                                          # tall rectangle (~3:4)
                                          0.5,
                                           1.0, 
                                           1.33,
                                           2,
                                           2.5# square
                                         # wide rectangle (~4:3)
                                          # very wide rectangle (3:1)
                                          # very tall rectangle (1:3)
                                       ]):
    """
    Generate patch sizes with multiple aspect ratios.

    Args:
        img_width (int): width of the image
        img_height (int): height of the image
        base_unit (int): smallest patch size unit (e.g., 32)
        num_scales (int): number of scales per aspect ratio
        min_fraction (float): min fraction of smaller dimension for smallest patch
        max_fraction (float): max fraction of smaller dimension for largest patch
        aspect_ratios (list of float): aspect ratios (width / height) to generate patches for

    Returns:
        List of tuples [(width1, height1), (width2, height2), ...] all multiples of base_unit
    """
    min_dim = min(img_width, img_height)
    min_size = int(min_dim * min_fraction)
    max_size = int(min_dim * max_fraction)

    # Round min/max sizes to nearest multiples of base_unit
    min_size = max(base_unit, (min_size // base_unit) * base_unit)
    max_size = (max_size // base_unit) * base_unit

    # Generate base square sizes linearly spaced between min and max
    base_sizes = []
    if num_scales > 1:
        step = (max_size - min_size) / (num_scales - 1)
    else:
        step = 0
    for i in range(num_scales):
        size = min_size + int(round(step * i))
        size = max(base_unit, round(size / base_unit) * base_unit)
        base_sizes.append(size)
    base_sizes = sorted(list(set(base_sizes)))

    all_patch_dimensions = []
    for aspect_ratio in aspect_ratios:
        for base_size in base_sizes:
            # Calculate width and height based on aspect ratio
            # aspect_ratio = width / height
            height = base_size / math.sqrt(aspect_ratio)
            width = base_size * math.sqrt(aspect_ratio)

            # Round width and height to multiples of base_unit
            width = max(base_unit, int(round(width / base_unit) * base_unit))
            height = max(base_unit, int(round(height / base_unit) * base_unit))

            # Avoid patches bigger than image dimensions
            if width <= img_width and height <= img_height:
                all_patch_dimensions.append((width, height))

    # Remove duplicates and sort by area descending (optional)
    all_patch_dimensions = list(set(all_patch_dimensions))
    all_patch_dimensions.sort(key=lambda x: x[0]*x[1], reverse=True)
    return all_patch_dimensions

# Initialize session state variables
if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None
if "last_uploaded_file_info" not in st.session_state:
    st.session_state.last_uploaded_file_info = None

# Sidebar for image upload and controls
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    # Process uploaded image
    if uploaded_file is not None:
        # Check if a new file has been uploaded or if it's the first run
        current_file_info = (uploaded_file.name, uploaded_file.size)
        if st.session_state.last_uploaded_file_info != current_file_info:
            st.session_state.last_uploaded_file_info = current_file_info

            # Load and process image
            st.info("Processing image... This may take a moment.")
            img = Image.open(uploaded_file).convert("RGB")
            width, height = img.size

            patch_dimensions = generate_patch_sizes_aspect_ratios(width, height)

            patches = []
            bboxes = []

            for (patch_w, patch_h) in patch_dimensions:
                # Overlap of 40%
                stride_x = int(patch_w * 0.6) if int(patch_w * 0.6) > 0 else 1
                stride_y = int(patch_h * 0.6) if int(patch_h * 0.6) > 0 else 1

                for y in range(0, height - patch_h + 1, stride_y):
                    for x in range(0, width - patch_w + 1, stride_x):
                        crop = img.crop((x, y, x + patch_w, y + patch_h))
                        bbox = (x, y, x + patch_w, y + patch_h)
                        patches.append(crop)
                        bboxes.append(bbox)
            
            # Compute embeddings for all patches
            image_embeddings = []
            for patch in patches:
                inputs = processor(images=patch, return_tensors="pt").to(device)
                with torch.no_grad():
                    img_feat = model.get_image_features(**inputs)
                    img_feat = img_feat.squeeze(0)  # remove batch dim
                image_embeddings.append(img_feat.cpu().numpy())
            
            if image_embeddings: # Ensure there are embeddings to stack
                image_embeddings = np.vstack(image_embeddings)
            else:
                image_embeddings = np.array([]) # Handle case with no patches

            st.session_state.uploaded_image_data = {
                "img": img,
                "patches": patches,
                "bboxes": bboxes,
                "image_embeddings": image_embeddings
            }
            st.success("Image processed and embeddings calculated!")
        
        # Always display the original image if available in session state
        if st.session_state.uploaded_image_data:
            st.image(st.session_state.uploaded_image_data["img"], caption="Original Image", use_column_width=True)
            st.write(f"Image dimensions: {st.session_state.uploaded_image_data['img'].width}x{st.session_state.uploaded_image_data['img'].height}")
            st.write(f"Number of patches generated: {len(st.session_state.uploaded_image_data['patches'])}")

# Main content area
st.header("Enter Your Query")
query = st.text_input("Describe the scene you are looking for:")
submit_button = st.button("Find Matching Scene")

if submit_button:
    if st.session_state.uploaded_image_data is None:
        st.warning("Please upload an image first and let it process.")
    elif not query:
        st.warning("Please enter a text query.")
    elif len(st.session_state.uploaded_image_data["patches"]) == 0:
        st.warning("No patches were generated from the image. Please try a different image or adjust patch generation parameters.")
    else:
        with st.spinner("Searching for the best match..."):
            # Compute text embedding
            inputs = processor(text=[query], return_tensors="pt").to(device)
            with torch.no_grad():
                text_embedding = model.get_text_features(**inputs)
            text_embedding = text_embedding.cpu().numpy()

            # Calculate similarities
            similarities = cosine_similarity(text_embedding, st.session_state.uploaded_image_data["image_embeddings"])
            best_match_idx = np.argmax(similarities)
            
            # Retrieve matched patch and its bounding box
            matched_patch = st.session_state.uploaded_image_data["patches"][best_match_idx]
            bbox = st.session_state.uploaded_image_data["bboxes"][best_match_idx]
            x1, y1, x2, y2 = bbox

            st.subheader("Results")
            
            # Display best matching patch
            st.write("Best matching patch:")
            st.image(matched_patch, caption=f"Matched Patch (Similarity: {similarities[0, best_match_idx]:.4f})", use_column_width=False)

            # Draw bounding box on original image
            img_with_box = st.session_state.uploaded_image_data["img"].copy()
            draw = ImageDraw.Draw(img_with_box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

            st.write("Original image with bounding box:")
            st.image(img_with_box, caption="Localized Scene", use_column_width=True)
            st.write(f"Bounding box coordinates (x1, y1, x2, y2): **({x1}, {y1}, {x2}, {y2})**")
            st.write(f"Similarity Score: **{similarities[0, best_match_idx]:.4f}**")