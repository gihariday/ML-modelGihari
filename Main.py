import os
import uvicorn
import traceback
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, Response
import json
import openai
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import re
import numpy as np
import random

# Initialize Model
# Load the trained model
model = load_model('chatbot_model.h5')

# Set up OpenAI API credentials
openai.api_key = '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'

# Load the intents from the JSON file
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

# Initialize tokenizer
tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(patterns)

# Prepare the training data
ptrn2seq = tokenizer.texts_to_sequences(patterns)
X = pad_sequences(ptrn2seq, padding='post')
lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(responses)

# This endpoint is for a test (or health check) to this server
app = FastAPI()

@app.get("/")
def index():
    return "Chatbot API"

class RequestText(BaseModel):
    text: str

def generate_dataset_response(user_input):
    for intent in intents:
        for pattern in intent['patterns']:
            if user_input.lower() == pattern.lower():
                return random.choice(intent['responses'])
    return None

def generate_chatgpt_response(user_input, chat_history):
    input_sequence = user_input.strip().lower()

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=chat_history + input_sequence,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None,
        )

        assistant_reply = response.choices[0].text.strip()
    except Exception as e:
        print("Error generating ChatGPT response:", e)
        assistant_reply = "I'm sorry, but I couldn't generate a response at the moment."
        response = None

    return assistant_reply, response


@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        # Text sent by the user
        user_input = req.text
        print("User Input:", user_input)

        # Generate response
        chat_history = "" 
        dataset_response = generate_dataset_response(user_input)
        if dataset_response is not None:
            response_text = dataset_response
        else:
            chatgpt_response, chatgpt_api_response = generate_chatgpt_response(user_input, chat_history)
            response_text = chatgpt_response
            print("ChatGPT API response:", chatgpt_api_response)

        # Return the response text
        return response_text
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


#Starting the server
port = int(os.environ.get("PORT", 8080))
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)