def predict_from_onnx(img):
    ort_session = ort.InferenceSession('membrane_segmentor.onnx')
    img = Image.fromarray(img)
    resized = img.resize((256, 256), Image.NEAREST)
    img_unsqueeze = expand_dims_twice(resized)
    onnx_outputs = ort_session.run(None, {'input': img_unsqueeze.astype('float32')})
    #ret = cv2.resize(onnx_outputs[0][0][0], (356,205), cv2.INTER_NEAREST)
    resized_ret = Image.fromarray(onnx_outputs[0][0][0]).resize((356, 256), Image.NEAREST)
    return(resized_ret)

def expand_dims_twice(arr):
    norm=(arr-np.min(arr))/(np.max(arr)-np.min(arr))
    ret = np.expand_dims(np.expand_dims(norm, axis=0), axis=0)
    return(ret)

import gradio as gr
import numpy as np
#import cv2
import onnxruntime as ort
from PIL import Image


ort_session = ort.InferenceSession('membrane_segmentor.onnx')
#img = cv2.imread("/content/devolearn/devolearn/tests/sample_data/images/seg_sample.jpg",0)
#examples=[img]

iface = gr.Interface(predict_from_onnx, 
                     gr.inputs.Image(image_mode="L", tool='edit'), 
                     gr.outputs.Image(),
                     title="GUI Demo - ONNX backend")
iface.launch(debug=False)