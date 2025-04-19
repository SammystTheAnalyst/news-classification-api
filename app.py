from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load the saved model and vectorizer
model = joblib.load("news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define a Welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the News Classification API"}

# Define a request model
class NewsRequest(BaseModel):
    text: str
# Define API route
@app.post("/predict/")
def predict_category(news: NewsRequest):
    # Preprocess the text
    processed_text = [news.text.lower()]

    # Convert to TF-IDF
    text_vectorized = vectorizer.transform(processed_text)

    # Predict category
    prediction = model.predict(text_vectorized)[0]

    return {"category": prediction}




