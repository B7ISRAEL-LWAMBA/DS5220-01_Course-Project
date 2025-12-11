# updated src/generator.py
import os
import json
import pandas as pd
import torch
from transformers import pipeline
from typing import List, Dict
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FewShotGenerator:
    def __init__(self, data_path: str = "data/processed/knowledge_base.csv"):
        """
        Initializes the Local HuggingFace Models and Knowledge Base.
        """
        print("Loading Knowledge Base...")
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            # Ensure text columns are strings
            self.df['full_transcript'] = self.df['full_transcript'].fillna("")
            self.df['company'] = self.df['company'].fillna("")
        else:
            self.df = pd.DataFrame(columns=["simulation_id", "company", "full_transcript"])

        print("Loading Local AI Models (This may take a moment)...")
        # Model 1: Categorization (Zero-Shot)
        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )
        
        # Model 2: Summarization (To extract steps)
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn"
        )
        
        self.categories = [
            "Insurance Claim", 
            "Payment Update", 
            "Order Status", 
            "Flight Booking", 
            "General Inquiry"
        ]

    def get_category(self, user_query: str) -> str:
        """
        Uses Bart-Large-MNLI to classify the user's intent.
        """
        result = self.classifier(user_query, candidate_labels=self.categories)
        return result['labels'][0]

    def find_best_match_transcript(self, user_query: str, category: str) -> str:
        """
        Finds the MOST SIMILAR transcript using TF-IDF (Keyword matching).
        This fixes the 'randomness' issue.
        """
        # 1. Filter by broad category first to narrow down the search
        subset = pd.DataFrame()
        
        if "Insurance" in category:
            subset = self.df[self.df['company'].str.contains("Insurance|Claim", case=False, na=False)]
        elif "Payment" in category:
            subset = self.df[self.df['company'].str.contains("Credit|Payment", case=False, na=False)]
        elif "Flight" in category:
            subset = self.df[self.df['company'].str.contains("Flight|Airline", case=False, na=False)]
        elif "Order" in category:
             subset = self.df[self.df['company'].str.contains("Order|Shipping|Retail", case=False, na=False)]
        
        # Fallback: If filter found nothing, search everything
        if subset.empty:
            subset = self.df
            
        if subset.empty:
            return ""

        # 2. Use TF-IDF to find the specific transcript that matches user keywords
        try:
            # Create a list containing the user query + all candidate transcripts
            documents = [user_query] + subset['full_transcript'].tolist()
            
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(documents)
            
            # Compute similarity of query (index 0) against all others
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # Get the index of the best match
            best_match_idx = cosine_sim.argmax()
            
            # Return that specific transcript
            return subset.iloc[best_match_idx]['full_transcript']
            
        except Exception as e:
            print(f"Similarity search failed: {e}. Falling back to random.")
            return subset.sample(1).iloc[0]['full_transcript']

    def generate_steps(self, user_query: str) -> Dict:
        """
        The Main Pipeline (Local Version):
        1. Predict Category.
        2. Find the BEST MATCH past case (Deterministic).
        3. Summarize that past case to suggest steps.
        """
        
        # 1. Identify Category
        category = self.get_category(user_query)
        
        # 2. Retrieve a similar historical case
        similar_transcript = self.find_best_match_transcript(user_query, category)
        
        # 3. Extract Steps/Summary using Local Summarizer
        steps_list = []
        reason = "Insufficient data."
        
        if similar_transcript:
            try:
                # Summarize the transcript to get the "Action Plan"
                # We focus on the middle part of the transcript where actions usually happen
                input_length = len(similar_transcript)
                start_idx = int(input_length * 0.15) 
                # Limit text to 1024 chars for the model
                text_to_summarize = similar_transcript[start_idx:start_idx+1024]
                
                # Generate abstractive summary
                summary_output = self.summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
                generated_plan = summary_output[0]['summary_text']
                
                # Split summary into a list based on sentences
                raw_steps = generated_plan.split('.')
                steps_list = [s.strip() for s in raw_steps if len(s) > 10]
                
                reason = f"Identified as {category}. Retrieved similar case logic."
            except Exception as e:
                steps_list = ["Could not extract specific steps from history."]
                reason = f"Error in local model: {e}"
        else:
            steps_list = ["No historical data found for this category."]
            reason = "No matching transcripts found in database."

        return {
            "category": category,
            "reason": reason,
            "steps": steps_list
        }