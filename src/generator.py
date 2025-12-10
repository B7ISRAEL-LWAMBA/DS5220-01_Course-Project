import os
import json
import pandas as pd
import torch
from transformers import pipeline
from typing import List, Dict
import random

class FewShotGenerator:
    def __init__(self, data_path: str = "data/processed/knowledge_base.csv"):
        """
        Initializes the Local HuggingFace Models.
        """
        print("Loading Knowledge Base...")
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
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
        # FIX: Ensure keyword argument is correct
        result = self.classifier(user_query, candidate_labels=self.categories)
        return result['labels'][0]

    def find_best_match_transcript(self, category: str) -> str:
        """
        Finds a transcript from the same category to use as a 'template'.
        """
        if "Insurance" in category:
            subset = self.df[self.df['company'].str.contains("Insurance|Claim", case=False, na=False)]
        elif "Payment" in category:
            subset = self.df[self.df['company'].str.contains("Credit|Payment", case=False, na=False)]
        elif "Flight" in category:
            subset = self.df[self.df['company'].str.contains("Flight|Airline", case=False, na=False)]
        else:
            subset = self.df
            
        if subset.empty:
            subset = self.df
            
        if not subset.empty:
            return subset.sample(1).iloc[0]['full_transcript']
        return ""

    def generate_steps(self, user_query: str) -> Dict:
        """
        The Main Pipeline (Local Version):
        1. Predict Category.
        2. Find a similar past case.
        3. Summarize that past case to suggest steps.
        """
        
        # 1. Identify Category
        category = self.get_category(user_query)
        
        # 2. Retrieve a similar historical case
        similar_transcript = self.find_best_match_transcript(category)
        
        # 3. Extract Steps/Summary using Local Summarizer
        if similar_transcript:
            try:
                # Summarize the transcript to get the "Action Plan"
                input_length = len(similar_transcript)
                start_idx = int(input_length * 0.2) 
                text_to_summarize = similar_transcript[start_idx:start_idx+1024]
                
                summary_output = self.summarizer(text_to_summarize, max_length=150, min_length=40, do_sample=False)
                generated_plan = summary_output[0]['summary_text']
                
                # Split summary into a list
                steps_list = [s.strip() for s in generated_plan.split('.') if len(s) > 10]
                reason = f"Identified as {category} based on keyword analysis."
            except Exception as e:
                steps_list = ["Could not extract specific steps from history."]
                reason = f"Error in local model: {e}"
        else:
            steps_list = ["No historical data found for this category."]
            reason = "Insufficient data."

        return {
            "category": category,
            "reason": reason,
            "steps": steps_list
        }