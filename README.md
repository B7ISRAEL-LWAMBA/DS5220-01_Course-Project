Sym Train Simulation Intelligence Assistant

A. Project Overview

The Sym Train Simulation Intelligence Assistant is an automated customer support pipeline designed to analyze historical simulation transcripts and assist agents in resolving customer requests.

Using Retrieval-Augmented Generation (RAG) and Few-Shot Learning, the system:

Ingests raw simulation data (JSON transcripts & Images).

Categorizes customer intent (e.g., Insurance Claims, Payment Updates).

Generates a step-by-step resolution plan based on similar historical successful cases.

Note: This project is architected to run 100% locally using open-source HuggingFace models (Bart-Large), ensuring zero data egress and no API costs for the main application.

Tech Stack & Dependencies

The project relies on a robust stack of Machine Learning and Web frameworks:

Frontend: Streamlit (Interactive Web UI)

NLP Engine: HuggingFace Transformers

Models:

facebook/bart-large-mnli (Zero-Shot Classification)

facebook/bart-large-cnn (Abstractive Summarization)

Data Processing: Pandas, NumPy

Deep Learning Backend: PyTorch, TensorFlow/Keras

Full Dependency List

Ensure these are installed via requirements.txt:

streamlit

pandas

torch

transformers

tf-keras

tensorflow

scikit-learn

openai

B. Installation & Setup

1. Prerequisite: Data Ingestion

The raw data provided by Sym Train is unstructured. You must run the data loader to clean, extract, and structure the data into a Knowledge Base.

Place your raw zip files into data/raw/.

Run the ETL script:

python src/data_loader.py


Result: A cleaned dataset will be created at data/processed/knowledge_base.csv.

2. Run the Intelligence App

Launch the web interface to interact with the AI assistant.

streamlit run app.py


The application will launch at http://localhost:8501.

First Run Note: The system will automatically download the necessary AI models (~3GB). This may take 1-3 minutes.

3. Run Model Analysis (Comparison Task)

To generate the performance comparison report (Local Transformer vs GPT-4o) for presentation slides:

Run the analysis script:

python src/analysis.py


Input API Key: The script will pause and prompt you to enter an OpenAI API Key.

Paste your key and press Enter to see the side-by-side comparison.

Press Enter without a key to run only the Local Model analysis.

C. Directory Structure

DS5220-01_Course-Project/
├── data/
│   ├── raw/                 # Source zip files (gitignored)
│   └── processed/           # knowledge_base.csv (Generated)
├── src/
│   ├── data_loader.py       # ETL Pipeline (Rescue & Repair logic)
│   ├── generator.py         # The AI "Brain" (Local RAG Pipeline)
│   └── analysis.py          # Model comparison script
├── app.py                   # Main Streamlit Application
├── Dockerfile               # Containerization configuration
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


D. Evaluation Test Cases

The system is validated against the following scenarios:

Payment Update (Amex): "Hi, I ordered a shirt last week and paid with my American Express card..."

Payment Update (General): "Hi, I need to update the payment method for one of my recent orders..."

Insurance Claim (Accident): "Hi, I am Sam. I was in a car accident this morning..."

Insurance Claim (Short): "Hi, can you help me file a claim?"

Order Status (General): "Hi, I recently ordered a book online..."

Order Status (Delay): "Hi, I have been waiting for two weeks for the book I ordered..."

F. Contributors

Developed for: DS5220 Course Project

Developed by: Israel Lwamba, Asiah Ruffin, and Eric

Supervised by: Professor Kong

Industry Partner: Sym Train