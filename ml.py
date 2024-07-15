import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pickle

# Load the data
tourist_df = pd.read_csv('edited.csv')

# Create a pipeline with TfidfVectorizer and MultinomialNB
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Define parameter grid for GridSearchCV
param_grid = {
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    'multinomialnb__alpha': [0.1, 0.5, 1.0, 2.0],
}

# Determine number of splits for cross-validation
city_counts = tourist_df['city'].value_counts()
min_samples = city_counts.min()
n_splits = min(5, min_samples)

# Perform GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=n_splits)
grid_search.fit(tourist_df['description'], tourist_df['city'])

# Define a function to recommend places based on user query
def recommend_places(query, threshold=0.01):
    predicted_probs = grid_search.predict_proba([query])[0]
    classes = grid_search.classes_
    city_prob_pairs = list(zip(classes, predicted_probs))
    sorted_city_prob_pairs = sorted(city_prob_pairs, key=lambda x: x[1], reverse=True)
    sorted_cities = [pair[0] for pair in sorted_city_prob_pairs if pair[1] >= threshold]
    return sorted_cities[:5]

# # Test the recommendation function
# query = "I like playing swimming fishing"
# recommended_cities = recommend_places(query)
# print("Recommended cities based on your query:")
# print(recommended_cities)

# Save the model to a pickle file
with open('tourist_recommendation_model.pkl', 'wb') as f:
    pickle.dump(grid_search, f)
