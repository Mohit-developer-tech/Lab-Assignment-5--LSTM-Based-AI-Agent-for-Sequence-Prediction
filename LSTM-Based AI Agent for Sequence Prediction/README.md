# Lab Assignment 5: LSTM-Based AI Agent for Sequence Prediction

## Group Members
- **Mohit Patil** (202301040272)
- **Parimal Ahire** (202301040067)
- **Rajveersinh Kher** (202301040233)
- **Atharva Suryawanshi** (202301040283)

---

## Project Overview
This project implements an **LSTM-based Next Word Prediction System** as part of the Deep Learning Lab. The system is trained on the Simple English Wikipedia dataset and deployed using **FastAPI** to create an industry-relevant AI agent.

### Key Features:
- **LSTM Sequence Learning**: Captures long-term dependencies in text for accurate context-aware predictions.
- **Data Visualization**: Includes pre-training word frequency analysis and post-training performance plots.
- **FastAPI Deployment**: Provides a production-ready API endpoint for real-time predictions.
- **Mathematical Modeling**: Implements and explains the internal gate mechanisms (Forget, Input, Output) of LSTMs.

---

## Technical Stack
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **API Framework**: FastAPI, Uvicorn
- **Data Analysis**: NumPy, Matplotlib, Seaborn
- **Notebook**: Jupyter Notebook (.ipynb)

---

## Getting Started

### 1. Installation
Install the required dependencies using pip:
```bash
pip install tensorflow fastapi uvicorn matplotlib seaborn requests
```

### 2. Training the Model
1. Open `LSTM_Sequence_Prediction.ipynb`.
2. Ensure the dataset path is correctly set to your local Wikipedia dataset.
3. Run all cells. This will:
   - Preprocess the text data.
   - Train the LSTM model.
   - Save the model as `lstm_next_word.h5` and the tokenizer as `tokenizer.pkl`.
   - Generate the `app.py` deployment script.

### 3. Running the API
Start the FastAPI server by running:
```bash
python -m uvicorn app:app --reload
```
The server will be available at `http://localhost:8000`.

---

## API Documentation

### Home Page
- **URL**: `http://localhost:8000/`
- **Method**: `GET`
- **Description**: Returns a status message and basic instructions.

### Next Word Prediction
- **URL**: `http://localhost:8000/predict`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "your seed text here",
    "num_words": 3
  }
  ```
- **Response**:
  ```json
  {
    "input": "your seed text here",
    "prediction": "your seed text here predicted words"
  }
  ```

### Interactive Docs
FastAPI provides automatic documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **Redoc**: `http://localhost:8000/redoc`

---

## Mathematical Model Summary
The LSTM cell manages information through three main gates:
- **Forget Gate ($f_t$)**: Decides what to drop from memory.
- **Input Gate ($i_t$)**: Decides what new info to store.
- **Output Gate ($o_t$)**: Decides what to output as the next hidden state.

This gating mechanism allows the model to remember context from the beginning of a sentence, which is essential for predicting the next word correctly.

---

