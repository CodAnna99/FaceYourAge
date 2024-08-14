# FaceYourAge
The goal of the project is to train a computer vision model on Edge Impulse to predict gender and age based on a person’s profile photo. This model will be integrated into a simple frontend with picture upload functionality, allowing for on-the-fly use without relying on on-premise installations.
The models were trained on approx. 23k images, all labeled with their gender (female/male) and age (1-116) with a 80/20 test split. 

## Ressources 

*   Edge Impulse CNN-Regression & Classification
*  [ Dataset UTK-Face ](https://www.kaggle.com/datasets/jangedoo/utkface-new) on Kaggel

*   [Ngrok Agent](https://ngrok.com/) for virtual server
*   [Flask Webframework](https://flask.palletsprojects.com/en/3.0.x/) for app rendering

# Step-by-Step Walkthrough 

## Install and Import Necessary Ressources
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

## Prepare Frontend Code 

The fronend contains a (dramatic) background picture, made with [leonardo.ai](https://www.leonardo.ai) and uses simple CSS and JavaScript elements to produce a picture upload area and the AI generated results after upload. 

![Face Your Age Frontend](https://github.com/CodAnna99/FaceYourAge/blob/main/faceyourageFE.jpg "Face Your Age")


## Load and Run Model

Set up interpreters and tensors for the use of the Edge Impulse Model. Make sure you use the prepared image from step above. 

### Model Performance Gender Classification


*   Accuracy: 86,44 %
*   Weighted average Precision: 0,90
*   Weighted average Recall: 0,90
*   Weighted average F1 score: 0,90

![confusionmatrix_gendermodel.jpg](https://github.com/CodAnna99/FaceYourAge/blob/main/confusionmatrix_gendermodel.jpg?raw=true)


### Model Performance Age Regression


*   Accuracy: 77,17 %
*   Mean squared error: 103,08
*   Mean absolute error: 7,75
*   Explained variance score: 0,74

## Starting App 

First render the frontend html to be used in the app. 
Then create a POST-function to the predict route created in load_and_run_model step, that returns the gender & age prediction. 

Two Edge Impulse models are used: CNN-Regression for age prediction and Classification for gender prediction. 
The models were deployed on Edge Impuls via a custom deployment block to extract the Tensorflow Lite Models: 

*   age_trained.tflite
*   gender_trained.tflite

This is especially useful as the models can be changed or retrained on EdgeImpulse any time and can still be used in this code as long as the models are named in the same schema and uploaded to the notebooks data storage. 


## Initializing ngrok Server


[ngrok](https://ngrok.com/) creates a virtual server to run the Flask app. It requires an authtoken to be used and generates a publicly available URL.
