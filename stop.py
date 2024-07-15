import nltk

# Specify the directory to store NLTK data
nltk_data_dir = 'nltk_data'

# Download necessary NLTK data to the specified directory
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
