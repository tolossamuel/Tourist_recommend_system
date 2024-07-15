import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
localtion = {
    "Addis-Ababa" : ["9.0192", "38.7525"],
    "Jimma" : ["7.6753", "36.8373"],
    "Axum" : ["14.1239", "38.7245"],
    "Awash National Park and Doho Hot Springs": ["9.13504", "40.08413"],
    "Gheralta" : ["13.8256", "39.2831"],
    "Alalobed Hot Springs" : [],
    "Arba Minch" : ["6.03333 N" , "37.55 E"],
    "Bahir Dar" : ["11.59364", "37.39077"],
    "Chebera Churchura National Park" : ["6.8872", "36.6363"],
    "Danakil Depression" : ["11.8764", "41.9196"],
    "Dire Dawa" : ["9.6049", "41.8585"],
    "Gambella National Park" : ["8.0046", "34.0641"],
    "Gondar" : ["12.6030", "37.4521"],
    "Hadar" : ["11.1932", "40.5999"],
    "Harar" : ["9.3124", "42.1218"],
    "Hawassa" : ["7.0477", "38.4958"],
    "Jinka" : ["5.7862", "36.5656"],
    "Kafa" : ["7.3042", "36.0893"],
    "Lalibela" : ["12.0309", "39.0476"],
    "Negash" :["13.8801", "39.5984"],
    "Simien Mountains" : ["13.4928", "38.4436"],
    "Wonchi" : ["8.666664", "37.916663"],
}
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
# Define a request body model
class TextInput(BaseModel):
    query: str

# Load the trained model
with open("tourist_recommendation_model.pkl", "rb") as f:
    classifier = pickle.load(f)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/predict')
def predict_city(query: str):
    words = word_tokenize(query)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    cleaned_query = ' '.join(filtered_words)
    predicted_probs = classifier.predict_proba([cleaned_query])[0]
    classes = classifier.classes_
    city_prob_pairs = list(zip(classes, predicted_probs))
    sorted_city_prob_pairs = sorted(city_prob_pairs, key=lambda x: x[1], reverse=True)
    
    sorted_cities = [[pair[0],localtion[pair[0].strip()]] for pair in sorted_city_prob_pairs if pair[1] >= 0.0]
    return sorted_cities[:10]

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)


'''
i like swimming and fishing

cultural festival in ethiopia

i like to visited rock mountain
i interest on desert volcano
i like fishing swimming and boat travel on lake

i like to old church cold weather condition and green land

small village and i wanna learn culture
'''