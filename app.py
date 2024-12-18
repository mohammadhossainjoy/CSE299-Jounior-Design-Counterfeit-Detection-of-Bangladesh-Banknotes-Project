#import library,package,framework,module
from flask import Flask, redirect, url_for, session, render_template, request, jsonify
from flask_oauthlib.client import OAuth
import numpy as np
import tensorflow as tf
import cv2
import os


#Flask App Start
app = Flask(__name__)
#secret key
app.secret_key = '5e884898da28047151d0e56f8dc62977'  
oauth = OAuth(app)

# This Configure of Google OAuth
google = oauth.remote_app(
    'google',
    # Client id and client secret from google cloud console
    consumer_key='1006252504341-ukl0s2m7l9ivttq5004a0na7h412mu71.apps.googleusercontent.com',
    consumer_secret='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',  
    request_token_params={ 'scope': 'email',},
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

# This code is Tensorflow(TFLite) model load
model_path = 'mobilenet_transfer_learning_model_for_counterfeit_detection_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# this is Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# uploaded image Pre-processing 
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize image
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

#page rendering start
@app.route('/')
def index():
    return render_template('login.html')  

@app.route('/home')
def home():
    if 'user' in session:
        #  index.html page render if user is logged in
        return render_template('index.html', user=session['user'])  
    # Redirect to user login but if not user  logged in
    return redirect(url_for('login'))  

@app.route('/login')
def login():
    # Redirect to login for Google
    return google.authorize(callback=url_for('authorized', _external=True))  

@app.route('/login/callback')
def authorized():
    response = google.authorized_response()
    if response is None or 'access_token' not in response:
        return 'Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        )
    
    session['google_token'] = (response['access_token'], '')
    # user information fetch from google
    user_info = google.get('userinfo')
    # user information store in session  
    session['user'] = user_info.data  
    # Redirect to home after user successful login
    return redirect(url_for('home'))  

@google.tokengetter
def get_google_oauth_token():
    # Retrieve the token from session
    return session.get('google_token')  

@app.route('/logout')
def logout():
    # remove the user from session
    session.pop('user', None) 
    # Remove google token from session 
    session.pop('google_token', None)
    # Redirect to login page  
    return redirect(url_for('index'))  


#This code start for model result prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded.'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'})

    # Save the uploaded image
    image_path = os.path.join('static', file.filename)
    file.save(image_path)

    # Image Preprocessing
    input_image = preprocess_image(image_path)

    # Set the tensor for input
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Get TensorFlow Lite models are lightweight and optimized for fast inference, especially in web apps.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)

    # Class mapping
    class_indices = {0: 'Your Bank note is Counterfeit', 1: 'Your Bank note is Original'}
    result = class_indices[predicted_class]

    # Remove the uploaded image after processing to save storage
    #Sends the prediction result back to the frontend as a JSON response.
    os.remove(image_path)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True) 