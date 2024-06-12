# Import necessary libraries
from transformers import pipeline

# Load the sentiment analysis pipeline
nlp_pipeline = pipeline("sentiment-analysis")

# Example texts
texts = [
    "I love this movie, it's fantastic!",
    "This product is terrible, I regret buying it."
]

# Perform sentiment analysis
for text in texts:
    # Get sentiment analysis result for each text
    result = nlp_pipeline(text)[0]
    
    # Print the text
    print(f"Text: {text}")
    
    # Print the sentiment label and score
    print(f"Sentiment: {result['label']}, Score: {result['score']}")
    
    # Print a blank line for better readability
    print()
  
