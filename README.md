# FaceYourAge
The goal of the project is to train a computer vision model on Edge Impulse to predict gender and age based on a person’s profile photo. This model will be integrated into a simple frontend with picture upload functionality, allowing for on-the-fly use without relying on on-premise installations.
The models were trained on approx. 23k images, all labeled with their gender (female/male) and age (1-116) with a 80/20 test split. 

Contributors: [@PhilippRamjoue](https://github.com/PhilippRamjoue ), [Philipp Harteneck](https://www.linkedin.com/in/philipp%2Dharteneck%2D1a682b2a/)



## Important Ressources 

*   Edge Impulse CNN-Regression & Classification
*  [ Dataset UTK-Face ](https://www.kaggle.com/datasets/jangedoo/utkface-new) on Kaggel

*   [Ngrok Agent](https://ngrok.com/) for virtual server
*   [Flask Webframework](https://flask.palletsprojects.com/en/3.0.x/) for app rendering

We used Google Colab for running the Jupyter notebook. 


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


## Models & Performance

Predictions are provides by two models trained on EdgeImpulse - one for Gender Classification and a CNN Regression for Age Prediction. Both models were optimized with the internal Edge Impulse EON-Tuner. The models were deployed on Edge Impuls via a custom deployment block to extract the Tensorflow Lite Models: 

*   age_trained.tflite
*   gender_trained.tflite

This is especially useful as the models can be changed or retrained on EdgeImpulse any time and can still be used in this code as long as the models are named in the same schema and uploaded to the notebooks data storage. 

### Model Performance Gender Classification


*   Accuracy: 86,44 %
*   Weighted average Precision: 0,90
*   Weighted average Recall: 0,90
*   Weighted average F1 score: 0,90

![confusionmatrix_gendermodel.jpg](https://github.com/CodAnna99/FaceYourAge/blob/main/confusionmatrix_gendermodel.jpg?raw=true)
![prediction_gendermodel](https://github.com/user-attachments/assets/740c7d1c-8ea8-48be-a8e7-6502553b0011)

Note: Gender Classification is especially prone to errors for very young faces. 


### Model Performance Age Regression


*   Accuracy: 77,17 %
*   Mean squared error: 103,08
*   Mean absolute error: 7,75
*   Explained variance score: 0,74
![prediction_agmodel](https://github.com/user-attachments/assets/55c177fe-4ea9-4bbb-8eba-30b8e6435104)

The CNN-Model uses 4 hidden layers and approx. 49k features to predict the age. 
![CNNR_agmodel](https://github.com/user-attachments/assets/f11c23ef-d88f-4203-a848-20a668bbedef)

Note: Age Prediction is notably blurry on the low and high end of the age scala - best results tend to be between 30-50 years of age.

## FaceYourAge-App 

Frontend is rendered within a Flask framework. It contains a (dramatic) background picture, made with [leonardo.ai](https://www.leonardo.ai) and uses simple CSS and JavaScript elements to produce a picture upload area and the AI generated results after upload. 

![Face Your Age Frontend](https://github.com/CodAnna99/FaceYourAge/blob/main/faceyourageFE.jpg "Face Your Age")
The image pload creates a POST request to the predict route created in 'load_and_run_model', that returns the gender & age prediction. 

[ngrok](https://ngrok.com/) creates a virtual server to run the Flask app. 
It requires an authtoken to be used and generates a publicly available URL.
