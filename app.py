import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import joblib

import numpy as np
import cv2
import onnxruntime as ort
import imutils
# import matplotlib.pyplot as plt
import pandas as pd

def onnx_segment_membrane(input_image):
    ort_session = ort.InferenceSession('membrane_segmentor.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    scaled_outputs = onnx_outputs[0][0][0]* 255
    resized_ret = Image.fromarray(scaled_outputs.astype(np.uint8) ).resize((356, 256), Image.NEAREST)#.convert("L")
    centroid_img = generate_centroid_image(np.array(onnx_outputs[0][0][0])) *255
    resized_centroid_img = Image.fromarray(centroid_img.astype(np.uint8)).resize((356, 256), Image.NEAREST)
    return(resized_ret, resized_centroid_img)

def generate_centroid_image(thresh):
    thresh = cv2.blur(thresh, (5,5))
    thresh = thresh.astype(np.uint8)
    centroid_image = np.zeros(thresh.shape)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    centroids = []
    for c in cnts:
        try:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            # cv2.drawContours(centroid_image, [c], -1, (255, 255, 255), 2)
            cv2.circle(centroid_image, (cX, cY), 2, (1, 1, 1), -1)
            centroids.append((cX, cY))
        except:
            pass
    return(centroid_image)

def onnx_segment_nucleus(input_image):
    ort_session = ort.InferenceSession('nucleus_segmentor.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    scaled_outputs = onnx_outputs[0][0][0]* 255
    resized_ret = Image.fromarray(scaled_outputs.astype(np.uint8) ).resize((708, 512), Image.NEAREST)#.convert("L")
    return(resized_ret)

def onnx_predict_lineage_population(input_image):
    ort_session = ort.InferenceSession('lineage_population_model.onnx')
    img = Image.fromarray(np.uint8(input_image))
    resized = img.resize((256, 256), Image.NEAREST)

    transposed=np.transpose(resized, (2, 1, 0))  
    img_unsqueeze = expand_dims(transposed)

    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')}) 
    return(onnx_outputs[0])


def expand_dims_twice(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    ret = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
    return(ret)

def expand_dims(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr)) #normalize
    ret = np.expand_dims(norm, axis=0)
    return(ret)

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def scale_model_outputs(scaler_path, data):
    scaler= joblib.load(scaler_path)
    scaled=scaler.inverse_transform(data)
    return(scaled)

def home():
    st.title('Home')
    #st.subheader('Markdown, images, charts, instructions, plotly plots')
    image = Image.open('images/banner_1.jpg')
    show = st.image(image, use_column_width=True)
    #st.text("https://github.com/Mainakdeb/devolearn/edit/master/README.md")
    intro_markdown = read_markdown_file("./home.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)

def cell_membrane_segmentation():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    ('Example_1.png','Example_2.png')
    )

    st.title('Cell Membrane Segmentation')
    instructions = """
        Segment Cell Membrane from C. elegans embryo imaging data \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/cell_membrane_segmentation_examples/'+selected_box2)

    col1, col2, col3 = st.beta_columns(3)

    if file:
        input = Image.open(file)
        col1.image(file, caption="input image")
    else:
        input = example_image
        
        col1.image(example_image, caption="input image")

    pressed = st.button('Run')
    if pressed:
        st.empty()
        col2.image(onnx_segment_membrane(np.array(input))[0], caption="segmentation map")
        col3.image(onnx_segment_membrane(np.array(input))[1], caption="centroid map")

def nucleus_segmentation():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    ('Example_1.png','Example_2.png')
    )

    st.title('Nucleus Segmentation')
    instructions = """
        Segment Nucleii from fluorescence microscopy imagery data (C. elegans embryo) \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/nucleus_segmentation_examples/'+selected_box2)

    col1, col2= st.beta_columns(2)

    if file:
        input = Image.open(file)
        col1.image(file, caption="input image")
    else:
        input = example_image
        
        col1.image(example_image, caption="input image")

    pressed = st.button('Run')
    if pressed:
        st.empty()
        col2.image(onnx_segment_nucleus(np.array(input)), caption="segmentation map")

def lineage_population_model():
    selected_box2 = st.sidebar.selectbox(
    'Choose Example Input',
    (['Example_1.png'])
    )

    st.title('Predict Cell Lineage Populations')
    instructions = """
        Predict the population of cells in C. elegans embryo using fluorescence microscopy data. \n
        Either upload your own image or select from the sidebar to get a preconfigured image. 
        The image you select or upload will be fed through the Deep Neural Network in real-time 
        and the output will be displayed to the screen.
        """
    st.text(instructions)
    file = st.file_uploader('Upload an image or choose an example')
    example_image = Image.open('./images/lineage_population_examples/'+selected_box2).convert("RGB")

    col1, col2= st.beta_columns(2)

    if file:
        input = Image.open(file).convert("RGB")
        col1.image(file, caption="input image")
    else:
        input = example_image
        
        col1.image(example_image, caption="input image")

    pressed = st.button('Run')
    if pressed:
        st.empty()
        output = onnx_predict_lineage_population(np.array(input))
        scaled_output = scale_model_outputs(scaler_path="./scaler.gz", data=output)

        for i in range(len(scaled_output[0])):
            scaled_output[0][i]=int(round(scaled_output[0][i]))

        df = pd.DataFrame({"Lineage":["A", "E", "M", "P", "C", "D", "Z"] , "Population": scaled_output[0]})
        col2.table(df)

st.set_page_config(page_title="DevoLearn", page_icon='🔬', layout='centered', initial_sidebar_state='auto')

hide_streamlit_style = """
            <style>
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Home','Cell Membrane Segmentation','Nucleus Segmentation', 'Predict Lineage populations')
    )

if selected_box == 'Home':
    home()
if selected_box == 'Cell Membrane Segmentation':
   cell_membrane_segmentation() 
if selected_box == 'Nucleus Segmentation':
    nucleus_segmentation() 
if selected_box == 'Predict Lineage populations':
    lineage_population_model()
