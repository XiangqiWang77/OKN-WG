from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained transformer model for embeddings
embedder = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased")

def get_embedding(text):
    # Obtain embedding using the pipeline
    embeddings = embedder(text)
    # Convert to numpy array and average the token embeddings to get a sentence embedding
    return np.mean(embeddings[0], axis=0)

def compute_similarity(user_input, json_data):
    user_embedding = get_embedding(user_input)
    category_similarities = []
    
    for category in json_data["question_categories"]:
        category_name = category["category"]
        category_embeddings = [get_embedding(q) for q in category["questions"]]
        
        # Compute the average similarity across all questions in the category
        similarities = [cosine_similarity([user_embedding], [qe])[0][0] for qe in category_embeddings]
        avg_similarity = np.mean(similarities)
        
        category_similarities.append((category_name, avg_similarity))
    
    # Find the category with the highest similarity
    best_match = max(category_similarities, key=lambda x: x[1])
    return best_match


import json

# Load JSON data from a file
with open('aspects.json', 'r') as f:
    json_data = json.load(f)


# Example user input
user_input = "Where can I observe alligators in Florida?"

# Compute the most similar category using the loaded JSON data
best_category, similarity_score = compute_similarity(user_input, json_data)
print(f"Best matching category: {best_category} with similarity score: {similarity_score:.2f}")

#Not practical given such example.