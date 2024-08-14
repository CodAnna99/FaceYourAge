# FaceYourAge
Project Goal The goal of the project is to train a computer vision model on Edge Impulse to predict gender and age based on a person’s profile photo. This model will be integrated into a simple frontend with picture upload functionality, allowing for on-the-fly use without relying on on-premise installations.
The models were trained on approx. 23k images, all labeled with their gender (female/male) and age (1-116) with a 80/20 test split. 

## Ressources 

*   Edge Impulse CNN-Regression & Classification
*  [ Dataset UTK-Face ](https://www.kaggle.com/datasets/jangedoo/utkface-new) on Kaggel

*   [Ngrok Agent](https://ngrok.com/) for virtual server
*   [Flask Webframework](https://flask.palletsprojects.com/en/3.0.x/) for app rendering

# Install and Import Necessary Ressources
```python
!pip install flask pyngrok
!pip install tflite-runtime


from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
import os
import base64
from io import BytesIO
from PIL import Image
from pyngrok import ngrok
```

# Prepare Frontend Code 

The fronend contains a (dramatic) background picture, made with [leonardo.ai](https://www.leonardo.ai) and uses simple CSS and JavaScript elements to produce a picture upload area and the AI generated results after upload. 

The frontend is in German.


