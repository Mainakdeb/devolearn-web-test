import gradio as gr
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import imutils

def predict_from_onnx(input_image):
    ort_session = ort.InferenceSession('membrane_segmentor.onnx')
    img = Image.fromarray(input_image)
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')})
    resized_ret = Image.fromarray(onnx_outputs[0][0][0]).resize((356, 256), Image.NEAREST)
    #centroid_img = generate_centroid_image(np.array(onnx_outputs[0][0][0]))
    #resized_centroid_img = Image.fromarray(centroid_img).resize((356, 256), Image.NEAREST)
    return(resized_ret)

# def generate_centroid_image(thresh):
#     thresh = cv2.blur(thresh, (5,5))
#     thresh = thresh.astype(np.uint8)
#     centroid_image = np.zeros(thresh.shape)
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     centroids = []
#     for c in cnts:
#         try:
#             # compute the center of the contour
#             M = cv2.moments(c)
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             # draw the contour and center of the shape on the image
#             # cv2.drawContours(centroid_image, [c], -1, (255, 255, 255), 2)
#             cv2.circle(centroid_image, (cX, cY), 2, (1, 1, 1), -1)
#             centroids.append((cX, cY))
#         except:
#             pass
#     return(centroid_image)

def expand_dims_twice(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    ret = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
    return(ret)

ort_session = ort.InferenceSession('membrane_segmentor.onnx')

iface = gr.Interface(predict_from_onnx, 
            gr.inputs.Image(image_mode="L"),
            [gr.outputs.Image(label="Segmentation Map"), gr.outputs.Image(label="Centroid Map")],
            title="DevoLearn - C. elegans Cell Membrane Segmentation",
            server_name="0.0.0.0")

iface.launch(debug=False)

