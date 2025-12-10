import os
import json
import zipfile
import shutil
import pandas as pd
from typing import List, Dict

# Constants
RAW_DIR = "data/raw"
TEMP_ZIP_DIR = "data/temp_zips"

def repair_directory_structure():
    """
    1. Moves zip files to safety.
    2. Wipes the cluttered 'data/raw' folder.
    3. Restores zips and extracts them into separate folders.
    """
    print(f"--- STARTING REPAIR OPERATION ---")
    
    # Check if raw directory exists
    if not os.path.exists(RAW_DIR):
        print(f"Error: {RAW_DIR} does not exist.")
        return False

    # 1. Identify and Rescue Zip Files
    files = os.listdir(RAW_DIR)
    zips = [f for f in files if f.endswith(".zip")]
    
    if not zips:
        print("No zip files found to rescue. Cannot proceed.")
        return False
        
    print(f"Rescuing {len(zips)} zip files to {TEMP_ZIP_DIR}...")
    
    if not os.path.exists(TEMP_ZIP_DIR):
        os.makedirs(TEMP_ZIP_DIR)
        
    for zip_file in zips:
        src = os.path.join(RAW_DIR, zip_file)
        dst = os.path.join(TEMP_ZIP_DIR, zip_file)
        shutil.move(src, dst)
        
    # 2. Wipe the Cluttered Directory
    print(f"Cleaning up clutter in {RAW_DIR}...")
    shutil.rmtree(RAW_DIR)
    os.makedirs(RAW_DIR)
    
    # 3. Restore and Extract
    print("Restoring and extracting files into proper folders...")
    for zip_file in zips:
        # Move zip back
        src = os.path.join(TEMP_ZIP_DIR, zip_file)
        dst_zip_path = os.path.join(RAW_DIR, zip_file)
        shutil.move(src, dst_zip_path)
        
        # Create a folder for this zip
        folder_name = zip_file.replace(".zip", "").strip()
        target_folder = os.path.join(RAW_DIR, folder_name)
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            try:
                with zipfile.ZipFile(dst_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)
                    print(f"✔ Extracted: {folder_name}")
            except zipfile.BadZipFile:
                print(f"Warning: {zip_file} is corrupt.")

    # Remove temp dir
    if os.path.exists(TEMP_ZIP_DIR):
        shutil.rmtree(TEMP_ZIP_DIR)
        
    print("--- REPAIR COMPLETE ---\n")
    return True

def load_simulation_data() -> List[Dict]:
    """
    Walks through the subfolders and finds the JSONs.
    """
    simulations = []
    print(f"Scanning {RAW_DIR} for JSON files...")
    
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith(".json"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                        
                        # Find the correct key (case-insensitive)
                        keys = {k.lower(): k for k in data.keys()}
                        audio_key = keys.get("audiocontentitems")
                        
                        if audio_key:
                            simulations.append({
                                "file_path": file_path,
                                "company": os.path.basename(os.path.dirname(file_path)),
                                "simulation_id": data.get("name", "unknown"),
                                "audio_items": data[audio_key]
                            })
                except Exception as e:
                    pass # Skip unreadable files
                    
    print(f"Found {len(simulations)} valid simulations.")
    return simulations

def process_transcripts(simulations: List[Dict]) -> pd.DataFrame:
    """
    Merges dialogue into a readable script.
    """
    processed_data = []

    for sim in simulations:
        audio_items = sim["audio_items"]
        full_transcript = []
        current_speaker = None
        current_block = []

        # Sort items
        audio_items.sort(key=lambda x: x.get('sequenceNumber', 0) if isinstance(x, dict) else 0)

        for item in audio_items:
            if not isinstance(item, dict): continue
            
            actor = item.get("actor", "Unknown")
            text = item.get("fileTranscript", "")
            
            if not text or not isinstance(text, str): continue
            text = text.strip()
            if not text: continue

            if actor != current_speaker:
                if current_speaker is not None:
                    full_transcript.append(f"{current_speaker}: {' '.join(current_block)}")
                current_speaker = actor
                current_block = [text]
            else:
                current_block.append(text)
        
        if current_speaker and current_block:
            full_transcript.append(f"{current_speaker}: {' '.join(current_block)}")
        
        processed_data.append({
            "simulation_id": sim["simulation_id"],
            "company": sim["company"],
            "full_transcript": "\n".join(full_transcript)
        })

    return pd.DataFrame(processed_data)

if __name__ == "__main__":
    # 1. Run Repair
    success = repair_directory_structure()
    
    if success:
        # 2. Load
        raw_sims = load_simulation_data()
        
        if raw_sims:
            # 3. Process
            df = process_transcripts(raw_sims)
            
            # 4. Save
            output_path = "data/processed/knowledge_base.csv"
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(output_path, index=False)
            
            print(f"\nSUCCESS! Processed {len(df)} simulations.")
            print(f"Data saved to: {output_path}")
        else:
            print("\nERROR: No simulations found even after repair.")