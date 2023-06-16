# README
# Hello everyone, in here I (Kaenova | Bangkit Mentor ML-20) 
# will give you some headstart on createing ML API. 
# Please read every lines and comments carefully. 
# 
# I give you a headstart on text based input and image based input API. 
# To run this server, don't forget to install all the libraries in the
# requirements.txt simply by "pip install -r requirements.txt" 
# and then use "python main.py" to run it
# 
# For ML:
# Please prepare your model either in .h5 or saved model format.
# Put your model in the same folder as this main.py file.
# You will load your model down the line into this code. 
# There are 2 option I give you, either your model image based input 
# or text based input. You need to finish functions "def predict_text" or "def predict_image"
# 
# For CC:
# You can check the endpoint that ML being used, eiter it's /predict_text or 
# /predict_image. For /predict_text you need a JSON {"text": "your text"},
# and for /predict_image you need to send an multipart-form with a "uploaded_file" 
# field. you can see this api documentation when running this server and go into /docs
# I also prepared the Dockerfile so you can easily modify and create a container iamge
# The default port is 8080, but you can inject PORT environement variable.
# 
# If you want to have consultation with me
# just chat me through Discord (kaenova#2859) and arrange the consultation time
#
# Share your capstone application with me! 🥳
# Instagram @kaenovama
# Twitter @kaenovama
# LinkedIn /in/kaenova

## Start your code here! ##

import os
import uvicorn
import traceback
import tensorflow as tf
from pydantic import BaseModel
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
import json
import openai
import pandas as pd
from keras.preprocessing.text import Tokenizer
import random
from keras.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import re
import numpy as np

# Initialize Model
# If you already put yout model in the same folder as this main.py
# You can load .h5 model or any model below this line

# If you use h5 type uncomment line below
# model = tf.keras.models.load_model('./my_model.h5')
# If you use saved model type uncomment line below
# model = tf.saved_model.load("./my_model_folder")

#prepare model dan API
app = FastAPI()
model = tf.keras.models.load_model('./1686713874.h5')
openai.api_key = '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'
with open('./intents.json', 'r') as f:
    data = json.load(f)

intents = data['intents']
df = pd.DataFrame(data['intents'])

patterns = []
responses = []

# Extract patterns and responses from intents
for intent in intents:
    patterns.extend(intent['patterns'])
    responses.extend(intent['responses'])

# Adjust the lengths of patterns and responses to match
min_len = min(len(patterns), len(responses))
patterns = patterns[:min_len]
responses = responses[:min_len]

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)

#initial tokenizer
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')
print('X shape = ', X.shape)

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])
print('y shape = ', y.shape)
print('num of classes = ', len(np.unique(y)))
X_shape =  (232, 18)
y_shape =  (232,)
num_of_classes =  80

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Chatbot Gihari"

# If your model need text input use this endpoint!
class RequestText(BaseModel):
    text:str

def generate_dataset_response(user_input): 
        text = []
        txt = re.sub('[^a-zA-Z\']', ' ', user_input)
        txt = txt.lower()
        txt = txt.split()
        txt = " ".join(txt)
        text.append(txt)
        
        x_test = tokenizer.texts_to_sequences(text)
        x_test = np.array(x_test).squeeze()
        x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
        y_pred = model.predict(x_test)
        y_pred = y_pred.argmax()
        tag = lbl_enc.inverse_transform([y_pred])[0]
        response = df[df['tag'] == tag]['responses'].values[0]
        responses = random.choice(response)
            
        return responses
def generate_chatgpt_response(user_input, chat_history):
        input_sequence = user_input.strip().lower()
        encoded_input = tokenizer.texts_to_sequences([input_sequence])[0]
        # Generate response from the OpenAI ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": chat_history},
                {"role": "user", "content": user_input}
                ],
                max_tokens=50
                )
        # Get the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()
        return assistant_reply   
        
@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # In here you will get text sent by the user
        chat_history = ""  # Initialize chat history
        print("Chatbot: How is your day?")
        user_input = req.text
        chat_history = ""  # Initialize chat history
        while True:
            user_input = input("User: ")  # Get user input
            # Find the matching pattern in the dataset
            matching_pattern = df[df['patterns'] == user_input]

            if matching_pattern is not None:
             # Get the responses for the matching pattern
            # Randomly select a response
                chatbot_reply = generate_dataset_response(user_input)
            else:
            # If no matching pattern is found, use the chatbot API
                chatbot_reply = generate_chatgpt_response(user_input, chat_history)

            print("Chatbot:", chatbot_reply)

             # Update chat history
            chat_history += f"User: {user_input}\nChatbot: {chatbot_reply}\n"
            if user_input.lower() == "Good bye":
                break
            
        return "Endpoint not implemented"
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
    return "Internal Server Error"

# Starting the server
# Your can check the API documentation easily using /docs after the server is running
port = os.environ.get("PORT", 8000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)