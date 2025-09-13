# requirements.txt
langchain==0.1.0
langchain-community==0.0.10
langchain-core==0.1.10
pydantic==2.5.0
tiktoken==0.5.2

# For local LLM support (optional - you can also use OpenAI/Anthropic APIs)
ollama==0.1.7

# For GUI
tkinter  # Usually comes with Python
typing-extensions==4.8.0

# Standard libraries (should be included with Python)
json
re
logging
pathlib
datetime
dataclasses
os

# setup_project.py
"""
Setup script for QA Dataset Generation Project
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "langchain==0.1.0",
        "langchain-community==0.0.10", 
        "langchain-core==0.1.10",
        "pydantic==2.5.0",
        "tiktoken==0.5.2"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
            return False
    
    return True

def setup_ollama():
    """Setup instructions for Ollama (optional local LLM)"""
    print("\n" + "="*50)
    print("OLLAMA SETUP (Optional - for local LLM)")
    print("="*50)
    print("If you want to use local LLMs instead of OpenAI/Anthropic APIs:")
    print("1. Install Ollama from: https://ollama.ai/")
    print("2. Run: ollama pull llama2")
    print("3. Or run: ollama pull llama2:13b for better quality")
    print("\nAlternatively, you can modify the code to use OpenAI or Anthropic APIs")

def create_project_structure():
    """Create project directory structure"""
    print("\nCreating project structure...")
    
    directories = [
        "datasets",
        "outputs",
        "logs",
        "temp"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def create_config_file():
    """Create configuration file"""
    config_content = """
# config.py
"""
Configuration file for QA Dataset Generation
"""

# LLM Configuration
LLM_CONFIG = {
    "provider": "ollama",  # Options: "ollama", "openai", "anthropic"
    "model_name": "llama2",  # For Ollama
    # "api_key": "your-api-key-here",  # For OpenAI/Anthropic
    "temperature": 0.7,
    "max_tokens": 2000
}

# Text Processing Configuration
TEXT_PROCESSING = {
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "min_chunk_size": 200,
    "max_chunk_size": 3000
}

# QA Generation Configuration
QA_GENERATION = {
    "questions_per_chunk": {
        "factual": 3,
        "procedural": 2, 
        "analytical": 2,
        "comparative": 1
    },
    "min_question_length": 10,
    "min_answer_length": 20,
    "quality_threshold": 0.7
}

# Output Configuration
OUTPUT = {
    "dataset_file": "outputs/kl_university_qa_dataset.json",
    "verified_file": "outputs/verified_qa_dataset.json",
    "logs_dir": "logs",
    "backup_dir": "outputs/backups"
}

# GUI Configuration
GUI = {
    "window_size": "1200x800",
    "auto_save_interval": 30,  # seconds
    "backup_frequency": 100   # every N edits
}
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("✓ Created config.py")

def create_example_script():
    """Create example usage script"""
    example_content = '''
# example_usage.py
"""
Example script showing how to use the QA Dataset Generator
"""

from qa_dataset_generator import UniversityQAGenerator
from qa_verification_gui import QAVerificationGUI
import tkinter as tk
import json

def generate_qa_dataset():
    """Generate QA dataset from university content"""
    print("Starting QA dataset generation...")
    
    # Initialize generator
    generator = UniversityQAGenerator(model_name="llama2")
    
    # Check if input file exists
    import os
    if not os.path.exists('all_content.txt'):
        print("Error: all_content.txt not found!")
        print("Please place your university content file in the same directory.")
        return
    
    # Load and process content
    print("Loading university content...")
    documents = generator.load_and_preprocess_content('all_content.txt')
    
    print("Chunking documents...")
    chunks = generator.chunk_documents(documents)
    
    print("Generating QA pairs...")
    qa_pairs = generator.generate_qa_pairs(chunks)
    
    # Save dataset
    output_file = 'outputs/kl_university_qa_dataset.json'
    generator.save_dataset(qa_pairs, output_file)
    
    # Generate statistics
    stats = generator.generate_summary_stats(qa_pairs)
    print("\\nDataset Generation Complete!")
    print(f"Generated {stats['total_pairs']} QA pairs")
    print(f"Average question length: {stats['avg_question_length']:.1f} characters")
    print(f"Average answer length: {stats['avg_answer_length']:.1f} characters")
    print(f"Saved to: {output_file}")
    
    return output_file

def launch_verification_gui(dataset_file=None):
    """Launch the verification GUI"""
    print("Launching QA Verification GUI...")
    
    root = tk.Tk()
    app = QAVerificationGUI(root)
    
    # Auto-load dataset if provided
    if dataset_file and os.path.exists(dataset_file):
        print(f"Auto-loading dataset: {dataset_file}")
        # You would need to add an auto-load method to the GUI class
    
    root.mainloop()

def main():
    """Main function - choose what to do"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "generate":
            dataset_file = generate_qa_dataset()
            
        elif command == "verify":
            dataset_file = None
            if len(sys.argv) > 2:
                dataset_file = sys.argv[2]
            launch_verification_gui(dataset_file)
            
        elif command == "both":
            # Generate then verify
            dataset_file = generate_qa_dataset()
            launch_verification_gui(dataset_file)
            
        else:
            print("Usage:")
            print("  python example_usage.py generate     # Generate QA dataset")
            print("  python example_usage.py verify       # Launch verification GUI")
            print("  python example_usage.py verify <file> # Launch GUI with dataset")
            print("  python example_usage.py both         # Generate then verify")
    else:
        print("What would you like to do?")
        print("1. Generate QA dataset")
        print("2. Launch verification GUI")
        print("3. Both (generate then verify)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            generate_qa_dataset()
        elif choice == "2":
            launch_verification_gui()
        elif choice == "3":
            dataset_file = generate_qa_dataset()
            launch_verification_gui(dataset_file)
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
'''
    
    with open("example_usage.py", "w") as f:
        f.write(example_content)
    
    print("✓ Created example_usage.py")

def main():
    """Main setup function"""
    print("QA Dataset Generation Project Setup")
    print("="*40)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install some requirements")
        return
    
    # Create project structure
    create_project_structure()
    
    # Create config file
    create_config_file()
    
    # Create example script
    create_example_script()
    
    # Setup instructions
    setup_ollama()
    
    print("\n" + "="*50)
    print("SETUP COMPLETE!")
    print("="*50)
    print("Next steps:")
    print("1. Place your 'all_content.txt' file in this directory")
    print("2. Run: python example_usage.py generate")
    print("3. Run: python example_usage.py verify")
    print("\nOr run both with: python example_usage.py both")
    print("\nFor more options, see example_usage.py")

if __name__ == "__main__":
    main()
