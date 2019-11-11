# tensorflow-js  
This repo implements semantic segmentation and style transfer on mobile devices. Images are taken from webcam. First we apply segmentation to source image(webcam image) and resize source image to give it to the style transfer model. After results of both models are received, style transfer results are upsampled back to initial size and applied on top of only pixels, which were classified as non-background.

# Models:  
* Semantic segmentation uses implementation of https://github.com/tensorflow/tfjs-models/tree/master/body-pix, which is Tensorflow-JS.  
* Style transfer model was converted from Pytorch into ONNX format and used in ONNX-JS using implementation found at https://github.com/gnsmrky/pytorch-fast-neural-style-for-web.   
# Launch:
* create python virtual environment and install requirments.txt
* Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1s6390jVpF-pBeb7Bvuekk8rxMqz-p_fA)
*