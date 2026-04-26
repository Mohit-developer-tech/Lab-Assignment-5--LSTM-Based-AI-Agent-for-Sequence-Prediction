
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import uvicorn

app = FastAPI(title='LSTM Next Word Prediction API')

@app.get('/')
def home():
    return {
        'message': 'LSTM Next Word Prediction API is running!',
        'instructions': 'Use the /predict endpoint (POST) to get predictions, or visit /docs for interactive documentation.'
    }

# Load model and tokenizer
model = tf.keras.models.load_model('lstm_next_word.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_sequence_len = model.input_shape[1] + 1

class PredictionRequest(BaseModel):
    text: str
    num_words: int = 1

@app.post('/predict')
def predict(request: PredictionRequest):
    seed_text = request.text
    for _ in range(request.num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += ' ' + output_word
    
    return {'input': request.text, 'prediction': seed_text}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
