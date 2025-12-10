import os

def inspect_files(start_path="data/raw"):
    print(f"--- INSPECTING {start_path} ---")
    
    file_count = 0
    extensions = {}
    
    for root, dirs, files in os.walk(start_path):
        for file in files:
            file_count += 1
            
            # Get extension
            ext = os.path.splitext(file)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1
            
            # Print the first 20 files found to give us a sample
            if file_count <= 20:
                print(f"Found: {os.path.join(root, file)}")
                
    print("\n--- SUMMARY ---")
    print(f"Total files found: {file_count}")
    print("File types found:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count}")

if __name__ == "__main__":
    inspect_files()