import streamlit as st
from PIL import Image
import numpy as np
import pixellib
from pixellib.semantic import semantic_segmentation
import os
import cv2


def config_image(image_file):
    img = Image.open(image_file)  # Streamlit Upload Type -> PIL Image Type
    # return img
    img.save('images/saved_images/{}'.format(image_file.name))


st.header("Semantic Segmentation and Texture Switching")

uploaded_file = st.sidebar.file_uploader('Upload your image', type=['jpg', 'jpeg', 'png'])
button = st.sidebar.button('Segment Image')

if uploaded_file is None and button:
    st.sidebar.warning('ERROR: Upload the image properly')

if button:
    with st.spinner("Processing..."):
        config_image(uploaded_file)

        segment_video = semantic_segmentation()
        segment_video.load_ade20k_model('models/deeplabv3_xception65_ade20k.h5')
        segment_video.segmentAsAde20k('images/saved_images/{}'.format(uploaded_file.name),
                                      output_image_name='images/output/{}'.format(uploaded_file.name))

        # Display input and output images
        image_col1, image_col2 = st.columns(2)
        with image_col1:
            st.header("Input Image")
            st.image('images/saved_images/{}'.format(uploaded_file.name))
        with image_col2:
            st.header("Output Image")
            st.image('images/output/{}'.format(uploaded_file.name))

        # Allow the user to select a segment (floor, wall, or ceiling) from the output image
        selected_segment = st.radio("Select a segment:", ('floor', 'wall', 'ceiling'))

        # Allow the user to upload a texture file to replace the selected segment
        texture_file = st.file_uploader('Upload a texture file', type=['jpg', 'jpeg', 'png'])

        # Check if a texture file has been uploaded
        if texture_file is not None:
            # Load the texture file
            texture_img = Image.open(texture_file)
            texture_img.save('images/{}'.format(texture_file.name))

            # Replace the selected segment with the texture
            if selected_segment == 'floor':
                with st.spinner("Processing..."):
                    # Perform color carpet operation
                    # Paste the texture onto the selected segment
                    output_img = Image.open('images/output/{}.jpg'.format(selected_segment))
                    output_img_arr = np.array(output_img)
                    mask = output_img_arr[:, :, 0] == 42
                    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    hull = cv2.convexHull(contour[0])
                    texture_img_arr = np.array(texture_img.resize(output_img.size))
                    texture_mask = np.zeros_like(mask)
                    cv2.fillPoly(texture_mask, [hull], 255)
                    texture_mask = texture_mask.astype(np.bool)[:, :, np.newaxis]
                    output_img_arr[texture_mask] = texture_img_arr[texture_mask]
                    output_img = Image.fromarray(output_img_arr)
                    output_img.save('images/output/{}_textured.jpg'.format(selected_segment.lower()))


            elif selected_segment == 'wall':
                with st.spinner("Processing..."):
                    # Perform color wall operation
                    # Paste the texture onto the selected segment
                    output_img = Image.open('images/output/{}.jpg'.format(selected_segment))
                    output_img_arr = np.array(output_img)
                    mask = output_img_arr[:, :, 0] == 8
                    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    hull = cv2.convexHull(contour[0])
                    texture_img_arr = np.array(texture_img.resize(output_img.size))
                    texture_mask = np.zeros_like(mask)
                    cv2.fillPoly(texture_mask, [hull], 255)
                    texture_mask = texture_mask.astype(np.bool)[:, :, np.newaxis]
                    output_img_arr[texture_mask] = texture_img_arr[texture_mask]
                    output_img = Image.fromarray(output_img_arr)
                    output_img.save('images/output/{}_textured.jpg'.format(selected_segment))

            elif selected_segment == 'ceiling':
                with st.spinner("Processing..."):
                    # Perform color ceiling operation
                    # Paste the texture onto the selected segment
                    output_img = Image.open('images/output/{}.jpg'.format(selected_segment))
                    output_img_arr = np.array(output_img)
                    mask = output_img_arr[:, :, 0] == 40
                    contour = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    hull = cv2.convexHull(contour[0])
                    texture_img_arr = np.array(texture_img.resize(output_img.size))
                    texture_mask = np.zeros_like(mask)
                    cv2.fillPoly(texture_mask, [hull], 255)
                    texture_mask = texture_mask.astype(np.bool)[:, :, np.newaxis]
                    output_img_arr[texture_mask] = texture_img_arr[texture_mask]
                    output_img = Image.fromarray(output_img_arr)
                    output_img.save('images/output/{}_textured.jpg'.format(selected_segment))

            # Show the updated image
            st.image(output_img, caption='Updated Image', use_column_width=True)

