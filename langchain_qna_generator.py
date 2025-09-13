#!/usr/bin/env python3
"""
LangChain-based QnA Dataset Generator with GUI Verification
Automatically detects and installs required libraries
"""

import sys
import subprocess
import importlib
import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for imported modules
RecursiveCharacterTextSplitter = None
HuggingFaceEmbeddings = None
FAISS = None
Document = None
OpenAI = None
ChatOpenAI = None
RetrievalQA = None
PromptTemplate = None
pd = None
np = None
st = None
load_dotenv = None

class LibraryManager:
    """Automatically detect and install required libraries"""
    
    REQUIRED_PACKAGES = {
        'langchain': 'langchain',
        'langchain_community': 'langchain-community',
        'langchain_openai': 'langchain-openai',
        'openai': 'openai',
        'tiktoken': 'tiktoken',
        'faiss': 'faiss-cpu',
        'sentence_transformers': 'sentence-transformers',
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'python-dotenv': 'python-dotenv'
    }
    
    @staticmethod
    def check_and_install_packages():
        """Check and install required packages"""
        missing_packages = []
        
        for module_name, package_name in LibraryManager.REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(module_name)
                logger.info(f"✓ {module_name} is already installed")
            except ImportError:
                missing_packages.append(package_name)
                logger.warning(f"✗ {module_name} is missing")
        
        if missing_packages:
            logger.info(f"Installing missing packages: {missing_packages}")
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"✓ Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"✗ Failed to install {package}: {e}")
                    raise
        
        # Import all required modules after installation
        return LibraryManager.import_modules()
    
    @staticmethod
    def import_modules():
        """Import all required modules and make them globally available"""
        try:
            # Make globals accessible
            global RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, FAISS, Document
            global OpenAI, ChatOpenAI, RetrievalQA, PromptTemplate, pd, np, st, load_dotenv
            
            # LangChain imports
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.docstore.document import Document
            from langchain_openai import ChatOpenAI
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            # Data processing
            import pandas as pd
            import numpy as np
            
            # GUI
            import streamlit as st
            
            # Environment
            from dotenv import load_dotenv
            
            logger.info("✓ All modules imported successfully")
            return True
            
        except ImportError as e:
            logger.error(f"✗ Failed to import modules: {e}")
            logger.error("Please ensure all required packages are installed:")
            for module_name, package_name in LibraryManager.REQUIRED_PACKAGES.items():
                logger.error(f"  pip install {package_name}")
            return False

class JSONDataProcessor:
    """Process and prepare JSON data for QnA generation"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.raw_data = self.load_json_data()
        self.documents = []
        
    def load_json_data(self) -> List[Dict]:
        """Load JSON data from file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    data = [data]
                logger.info(f"✓ Loaded {len(data)} items from JSON file")
                return data
        except Exception as e:
            logger.error(f"✗ Error loading JSON: {e}")
            return []
    
    def create_documents(self) -> List[Document]:
        """Convert JSON data to LangChain Documents"""
        if Document is None:
            raise ImportError("Document class not imported. Please run LibraryManager.check_and_install_packages() first.")
            
        documents = []
        
        for idx, item in enumerate(self.raw_data):
            # Extract text content from various fields
            content_parts = []
            metadata = {"source": self.json_file_path, "index": idx}
            
            # Common content fields
            content_fields = ['content', 'description', 'text', 'body', 'summary']
            title_fields = ['title', 'name', 'subject', 'heading']
            
            # Extract title
            title = None
            for field in title_fields:
                if field in item and item[field]:
                    title = str(item[field])
                    metadata['title'] = title
                    break
            
            # Extract main content
            for field in content_fields:
                if field in item and item[field]:
                    content_parts.append(str(item[field]))
            
            # Add other string fields as additional context
            for key, value in item.items():
                if key not in content_fields + title_fields and isinstance(value, (str, int, float)):
                    if len(str(value)) > 10:  # Only meaningful content
                        content_parts.append(f"{key}: {value}")
                    metadata[key] = value
            
            # Combine content
            if title:
                full_content = f"Title: {title}\n\n" + "\n".join(content_parts)
            else:
                full_content = "\n".join(content_parts)
            
            if full_content.strip():
                doc = Document(page_content=full_content, metadata=metadata)
                documents.append(doc)
        
        logger.info(f"✓ Created {len(documents)} documents")
        self.documents = documents
        return documents

class LangChainQnAGenerator:
    """Generate QnA pairs using LangChain and LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.setup_components()
    
    def setup_components(self):
        """Setup LangChain components"""
        try:
            # Check if modules are imported
            if HuggingFaceEmbeddings is None:
                raise ImportError("Required modules not imported. Please run LibraryManager.check_and_install_packages() first.")
            
            # Setup embeddings (using free HuggingFace embeddings)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Setup LLM (try OpenAI first, fallback to local if no key)
            if self.api_key and ChatOpenAI is not None:
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.1
                )
                logger.info("✓ Using OpenAI GPT-3.5-turbo")
            else:
                logger.warning("⚠ No OpenAI API key found. Using template-based generation.")
                self.llm = None
            
        except Exception as e:
            logger.error(f"✗ Error setting up components: {e}")
    
    def create_vectorstore(self, documents: List[Document]):
        """Create vector store from documents"""
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            
            logger.info(f"✓ Created vectorstore with {len(split_docs)} chunks")
            
        except Exception as e:
            logger.error(f"✗ Error creating vectorstore: {e}")
    
    def generate_questions_from_content(self, content: str, title: str = "") -> List[str]:
        """Generate questions from content using templates or LLM"""
        questions = []
        
        if self.llm:
            # Use LLM to generate questions
            prompt = f"""
            Based on the following content, generate 3-5 diverse questions that can be answered using this information.
            Make the questions specific, clear, and varied in type (factual, explanatory, comparative, etc.).
            
            Title: {title}
            Content: {content[:1500]}...
            
            Generate questions only, one per line:
            """
            
            try:
                response = self.llm.invoke(prompt)
                # Handle different response types
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                questions = [q.strip() for q in response_text.split('\n') if q.strip() and '?' in q]
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, falling back to templates")
                questions = self.generate_template_questions(content, title)
        else:
            # Use template-based generation
            questions = self.generate_template_questions(content, title)
        
        return questions[:5]  # Limit to 5 questions
    
    def generate_template_questions(self, content: str, title: str = "") -> List[str]:
        """Generate questions using templates"""
        questions = []
        subject = title if title else "this topic"
        
        # Question templates
        templates = [
            f"What is {subject}?",
            f"How does {subject} work?",
            f"What are the key features of {subject}?",
            f"Why is {subject} important?",
            f"What should I know about {subject}?",
            f"Can you explain {subject}?",
            f"What are the benefits of {subject}?",
            f"How can {subject} be used?"
        ]
        
        # Select relevant templates based on content
        for template in templates[:5]:
            questions.append(template)
        
        return questions
    
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer for a question using context"""
        if self.llm and self.retriever:
            try:
                # Use retrieval QA
                qa_prompt = PromptTemplate(
                    template="""Use the following context to answer the question. If you cannot answer based on the context, say "I don't have enough information to answer this question."
                    
                    Context: {context}
                    Question: {question}
                    Answer:""",
                    input_variables=["context", "question"]
                )
                
                # Get relevant context
                docs = self.retriever.invoke(question)
                context_text = "\n".join([doc.page_content for doc in docs])
                
                # Generate answer
                formatted_prompt = qa_prompt.format(context=context_text, question=question)
                response = self.llm.invoke(formatted_prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
                
                return answer.strip()
                
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}")
        
        # Fallback: extract relevant sentences from context
        return self.extract_answer_from_context(question, context)
    
    def extract_answer_from_context(self, question: str, context: str) -> str:
        """Extract answer from context using simple heuristics"""
        sentences = context.split('. ')
        
        # Simple keyword matching
        question_words = question.lower().split()
        best_sentences = []
        
        for sentence in sentences:
            score = sum(1 for word in question_words if word in sentence.lower())
            if score > 0:
                best_sentences.append((sentence, score))
        
        if best_sentences:
            best_sentences.sort(key=lambda x: x[1], reverse=True)
            return best_sentences[0][0] + '.'
        
        return "Based on the available information: " + sentences[0][:200] + "..."
    
    def generate_qna_dataset(self, documents: List[Document]) -> List[Dict]:
        """Generate complete QnA dataset"""
        qna_pairs = []
        
        # Create vectorstore
        self.create_vectorstore(documents)
        
        for doc in documents:
            title = doc.metadata.get('title', '')
            content = doc.page_content
            
            # Generate questions
            questions = self.generate_questions_from_content(content, title)
            
            # Generate answers for each question
            for question in questions:
                answer = self.generate_answer(question, content)
                
                qna_pair = {
                    'question': question,
                    'answer': answer,
                    'context': content[:500] + "..." if len(content) > 500 else content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'metadata': doc.metadata
                }
                qna_pairs.append(qna_pair)
        
        logger.info(f"✓ Generated {len(qna_pairs)} QnA pairs")
        return qna_pairs

def main():
    """Main execution function"""
    # Check and install required packages
    print("Checking and installing required packages...")
    if not LibraryManager.check_and_install_packages():
        print("Failed to setup required packages. Exiting.")
        return
    
    # Load environment variables
    load_dotenv()
    
    print("\n" + "="*60)
    print("LangChain QnA Dataset Generator")
    print("="*60)
    
    # Get input file
    json_file = input("Enter path to your JSON file: ").strip()
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found!")
        return
    
    # Get OpenAI API key (optional)
    api_key = input("Enter OpenAI API key (optional, press Enter to skip): ").strip()
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    print("\nProcessing your data...")
    
    # Process JSON data
    processor = JSONDataProcessor(json_file)
    documents = processor.create_documents()
    
    if not documents:
        print("No valid documents found in JSON file!")
        return
    
    # Generate QnA dataset
    generator = LangChainQnAGenerator(api_key)
    qna_dataset = generator.generate_qna_dataset(documents)
    
    if not qna_dataset:
        print("Failed to generate QnA dataset!")
        return
    
    # Save initial dataset
    output_file = json_file.replace('.json', '_qna_dataset.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qna_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(qna_dataset)} QnA pairs")
    print(f"✓ Saved initial dataset to: {output_file}")
    print(f"\nStarting GUI for verification and editing...")
    
    # Launch Streamlit GUI
    gui_file = "qna_verification_gui.py"
    create_streamlit_gui(gui_file, output_file)
    
    print(f"\nGUI created: {gui_file}")
    print("Run the following command to start the GUI:")
    print(f"streamlit run {gui_file}")

def create_streamlit_gui(gui_file: str, dataset_file: str):
    """Create Streamlit GUI for QnA verification"""
    gui_code = f'''
import streamlit as st
import json
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="QnA Dataset Verification",
    page_icon="📝",
    layout="wide"
)

DATASET_FILE = "{dataset_file}"

@st.cache_data
def load_dataset():
    """Load QnA dataset"""
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading dataset: {{e}}")
        return []

def save_dataset(dataset):
    """Save QnA dataset"""
    try:
        # Create backup
        backup_file = DATASET_FILE.replace('.json', f'_backup_{{datetime.now().strftime("%Y%m%d_%H%M%S")}}.json')
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save updated dataset
        with open(DATASET_FILE, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        st.success(f"✓ Dataset saved! Backup created: {{backup_file}}")
        return True
    except Exception as e:
        st.error(f"Error saving dataset: {{e}}")
        return False

def main():
    st.title("📝 QnA Dataset Verification & Editing")
    st.markdown("Review and edit your generated QnA pairs to ensure accuracy.")
    
    # Load dataset
    dataset = load_dataset()
    if not dataset:
        st.error("No dataset found or failed to load!")
        return
    
    # Sidebar for navigation and stats
    with st.sidebar:
        st.header("📊 Dataset Statistics")
        st.metric("Total QnA Pairs", len(dataset))
        
        # Filter options
        st.header("🔧 Filters")
        show_all = st.checkbox("Show all pairs", value=True)
        
        if not show_all:
            source_filter = st.selectbox(
                "Filter by source",
                options=["All"] + list(set(item.get('source', 'unknown') for item in dataset))
            )
        else:
            source_filter = "All"
        
        # Search
        search_term = st.text_input("🔍 Search questions", "")
    
    # Filter dataset
    filtered_dataset = dataset.copy()
    if source_filter != "All":
        filtered_dataset = [item for item in filtered_dataset if item.get('source', 'unknown') == source_filter]
    
    if search_term:
        filtered_dataset = [
            item for item in filtered_dataset 
            if search_term.lower() in item['question'].lower()
        ]
    
    st.write(f"Showing {{len(filtered_dataset)}} of {{len(dataset)}} QnA pairs")
    
    # Pagination
    items_per_page = st.selectbox("Items per page", [5, 10, 20, 50], index=1)
    total_pages = (len(filtered_dataset) - 1) // items_per_page + 1 if filtered_dataset else 0
    
    if total_pages > 0:
        page = st.number_input("Page", 1, total_pages, 1) - 1
        start_idx = page * items_per_page
        end_idx = start_idx + items_per_page
        page_items = filtered_dataset[start_idx:end_idx]
        
        # Display QnA pairs
        for i, item in enumerate(page_items):
            actual_idx = dataset.index(item)  # Get original index
            
            with st.expander(f"QnA Pair {{start_idx + i + 1}}: {{item['question'][:60]}}..."):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("❓ Question")
                    new_question = st.text_area(
                        "Edit question:",
                        value=item['question'],
                        key=f"q_{{actual_idx}}",
                        height=100
                    )
                
                with col2:
                    st.subheader("💬 Answer")
                    new_answer = st.text_area(
                        "Edit answer:",
                        value=item['answer'],
                        key=f"a_{{actual_idx}}",
                        height=100
                    )
                
                # Context and metadata
                st.subheader("📄 Context")
                st.text_area(
                    "Context (read-only):",
                    value=item.get('context', ''),
                    key=f"c_{{actual_idx}}",
                    height=80,
                    disabled=True
                )
                
                # Action buttons
                col3, col4, col5 = st.columns([1, 1, 1])
                
                with col3:
                    if st.button(f"✅ Update", key=f"update_{{actual_idx}}"):
                        dataset[actual_idx]['question'] = new_question
                        dataset[actual_idx]['answer'] = new_answer
                        st.success("Updated!")
                        st.rerun()
                
                with col4:
                    if st.button(f"🗑️ Delete", key=f"delete_{{actual_idx}}"):
                        dataset.pop(actual_idx)
                        st.success("Deleted!")
                        st.rerun()
                
                with col5:
                    quality_score = st.selectbox(
                        "Quality",
                        options=["Good", "Needs Review", "Poor"],
                        index=0,
                        key=f"quality_{{actual_idx}}"
                    )
                    dataset[actual_idx]['quality'] = quality_score
        
        # Global actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Save All Changes", type="primary"):
                if save_dataset(dataset):
                    st.rerun()
        
        with col2:
            if st.button("📥 Download Dataset"):
                st.download_button(
                    label="📄 Download JSON",
                    data=json.dumps(dataset, indent=2, ensure_ascii=False),
                    file_name=f"verified_qna_dataset_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("➕ Add New QnA"):
                st.session_state.show_add_form = True
        
        # Add new QnA form
        if st.session_state.get('show_add_form', False):
            st.markdown("---")
            st.subheader("➕ Add New QnA Pair")
            
            new_q = st.text_input("Question:")
            new_a = st.text_area("Answer:")
            new_context = st.text_area("Context (optional):")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Add QnA"):
                    if new_q and new_a:
                        dataset.append({{
                            'question': new_q,
                            'answer': new_a,
                            'context': new_context,
                            'source': 'manual_entry',
                            'metadata': {{'added_manually': True}}
                        }})
                        st.success("Added new QnA pair!")
                        st.session_state.show_add_form = False
                        st.rerun()
                    else:
                        st.error("Please fill in both question and answer!")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.show_add_form = False
                    st.rerun()

if __name__ == "__main__":
    main()
'''
    
    with open(gui_file, 'w', encoding='utf-8') as f:
        f.write(gui_code)

if __name__ == "__main__":
    main()