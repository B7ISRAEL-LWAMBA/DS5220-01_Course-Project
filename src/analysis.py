# Final src/analysis.py
import os
import json
import pandas as pd
# Suppress TensorFlow logs to keep output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline
from openai import OpenAI
from typing import List, Dict

class AnalysisEngine:
    def __init__(self):
        print("Loading Local Models...")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.candidate_labels = ["Insurance Claim", "Payment Update", "Order Status", "Flight Booking", "General Inquiry"]

    def analyze_local(self, text: str) -> Dict:
        """Run Local Transformer Analysis"""
        # 1. Categorize
        cat_result = self.classifier(text[:1024], candidate_labels=self.candidate_labels)
        
        # 2. Summarize steps
        try:
            summary = self.summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)
            steps_text = summary[0]['summary_text']
        except:
            steps_text = "Error in summarization."
            
        return {
            "category": cat_result['labels'][0], 
            "confidence": cat_result['scores'][0], 
            "steps": steps_text
        }

    def analyze_gpt(self, text: str, api_key: str) -> Dict:
        """Run GPT-4o Analysis"""
        client = OpenAI(api_key=api_key)
        prompt = f"""
        Analyze this customer service transcript:
        "{text[:2000]}"
        
        1. Categorize into one of: {self.candidate_labels}
        2. Extract the exact steps taken by the agent.
        
        Return JSON: {{ "category": "...", "steps": ["step1", "step2"] }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}], 
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

def run_comparison():
    data_path = "data/processed/knowledge_base.csv"
    if not os.path.exists(data_path):
        print("Data file not found! Please run data_loader.py first.")
        return

    print("\n" + "="*60)
    print("  API KEY CHECK")
    print("To run the comparison, please paste your OpenAI API Key below.")
    print("If you don't have one, just press ENTER to see only the Local results.")
    api_key = input("API Key > ").strip()
    
    df = pd.read_csv(data_path)
    engine = AnalysisEngine()

    print("\n" + "="*60)
    print("  MODEL COMPARISON: LOCAL vs. GPT (3 SAMPLES)")
    print("="*60)

    # Run on 3 random samples for a solid evaluation
    if not df.empty:
        # Use sample(n) where n is min(3, len(df)) in case df is small
        n_samples = min(3, len(df))
        samples = df.sample(n_samples)
        
        for i, (index, row) in enumerate(samples.iterrows()):
            print(f"\n--- SAMPLE {i+1}: {row['simulation_id']} ---")
            
            # Local
            print(" [Local Model]")
            local_res = engine.analyze_local(row['full_transcript'])
            print(f"   Category: {local_res['category']} (Conf: {local_res['confidence']:.2f})")
            print(f"   Summary:  {local_res['steps'][:150]}...") # Truncate for clean output

            # GPT
            if api_key:
                print(" [GPT-4o]")
                gpt_res = engine.analyze_gpt(row['full_transcript'], api_key)
                
                # --- NEW DEBUGGING LINES ---
                if "error" in gpt_res:
                    print(f"   ERROR: {gpt_res['error']}")
                else:
                    print(f"   Category: {gpt_res.get('category')}")
                    print(f"   Steps:    {gpt_res.get('steps')}")
            else:
                print(" [GPT-4o] Skipped")
            
            print("-" * 40)

if __name__ == "__main__":
    run_comparison()