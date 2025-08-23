import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import sys
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Any
import re

# Automatic dependency installation
def install_dependencies():
    """Install required packages for GUI and data handling"""
    packages = [
        "pandas",
        "pillow",  # For enhanced GUI features
    ]
    
    for package in packages:
        try:
            if package == "pillow":
                import PIL
            else:
                __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

# Install dependencies
try:
    install_dependencies()
    import pandas as pd
except Exception as e:
    print(f"Error installing dependencies: {e}")
    sys.exit(1)

class DatasetStructurer:
    """
    Phase 3: Enhanced dataset structuring with metadata
    """
    
    def __init__(self):
        self.dataset_schema = {
            "id": "string",
            "question": "string",
            "answer": "string",
            "category": "string",
            "sub_category": "string",
            "source_page": "integer",
            "source_title": "string",
            "source_url": "string",
            "confidence": "float",
            "question_type": "string",
            "difficulty_level": "string",
            "keywords": "list",
            "created_date": "string",
            "modified_date": "string",
            "review_status": "string",
            "reviewer_notes": "string",
            "generation_method": "string",
            "validation_score": "float"
        }
        
        self.question_types = [
            "factual",           # Direct factual questions
            "procedural",        # How-to questions
            "comparative",       # Comparison questions
            "numerical",         # Number-based questions
            "yes_no",           # Yes/No questions
            "multiple_choice",   # Multiple choice format
            "descriptive",       # Descriptive explanations
            "list_based"        # List/enumeration questions
        ]
        
        self.difficulty_levels = [
            "basic",        # Simple, direct questions
            "intermediate", # Moderate complexity
            "advanced",     # Complex, analytical questions
            "expert"        # Deep domain knowledge required
        ]
        
        self.review_statuses = [
            "unreviewed",   # Initial state
            "approved",     # Approved for training
            "needs_edit",   # Requires modifications
            "rejected",     # Not suitable for training
            "flagged"       # Flagged for special attention
        ]
    
    def generate_unique_id(self, prefix: str = "qa") -> str:
        """Generate unique identifier for Q&A pair"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_suffix}"
    
    def classify_question_type(self, question: str) -> str:
        """Automatically classify question type based on content"""
        question_lower = question.lower().strip()
        
        # Pattern-based classification
        if question_lower.startswith(("what is", "what are", "who is", "where is")):
            return "factual"
        elif question_lower.startswith(("how to", "how can", "how do")):
            return "procedural"
        elif "compare" in question_lower or "difference" in question_lower:
            return "comparative"
        elif re.search(r'\bhow many\b|\bhow much\b|\bwhat.*number\b', question_lower):
            return "numerical"
        elif question_lower.startswith(("is", "are", "does", "do", "can", "will")):
            return "yes_no"
        elif "list" in question_lower or "what are the" in question_lower:
            return "list_based"
        else:
            return "descriptive"
    
    def assess_difficulty_level(self, question: str, answer: str) -> str:
        """Assess difficulty level based on question and answer complexity"""
        # Simple metrics for difficulty assessment
        question_words = len(question.split())
        answer_words = len(answer.split())
        
        # Keywords indicating complexity
        complex_keywords = ["analysis", "compare", "evaluate", "strategy", "implementation", 
                          "framework", "methodology", "research", "advanced", "specialized"]
        
        basic_keywords = ["what", "when", "where", "name", "list", "fee", "duration"]
        
        complexity_score = 0
        
        # Length-based scoring
        if question_words > 15 or answer_words > 100:
            complexity_score += 2
        elif question_words > 10 or answer_words > 50:
            complexity_score += 1
        
        # Keyword-based scoring
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        for keyword in complex_keywords:
            if keyword in question_lower or keyword in answer_lower:
                complexity_score += 1
        
        for keyword in basic_keywords:
            if keyword in question_lower:
                complexity_score -= 1
        
        # Classify based on score
        if complexity_score <= 0:
            return "basic"
        elif complexity_score <= 2:
            return "intermediate"
        elif complexity_score <= 4:
            return "advanced"
        else:
            return "expert"
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction
        import re
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why'}
        
        # Extract words (alphanumeric, 3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and get frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_keywords]]
    
    def calculate_validation_score(self, question: str, answer: str, confidence: float) -> float:
        """Calculate validation score based on various quality metrics"""
        score = 0.0
        
        # Question quality metrics
        if len(question) >= 10:
            score += 0.2
        if '?' in question:
            score += 0.2
        if len(question.split()) >= 5:
            score += 0.1
        
        # Answer quality metrics
        if len(answer) >= 20:
            score += 0.2
        if len(answer) <= 500:  # Not too long
            score += 0.1
        if answer.endswith('.'):
            score += 0.1
        
        # Confidence factor
        score += confidence * 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def structure_dataset(self, input_file: str, output_file: str) -> Dict:
        """Main function to structure the dataset with enhanced metadata"""
        print("Loading dataset from Phase 2...")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: {input_file} not found. Please run Phase 2 first.")
            return None
        
        print("Structuring dataset with enhanced metadata...")
        
        structured_questions = []
        
        for i, question_data in enumerate(raw_data.get('questions', [])):
            # Generate unique ID
            unique_id = self.generate_unique_id()
            
            # Extract basic information
            question = question_data.get('question', '').strip()
            answer = question_data.get('answer', '').strip()
            category = question_data.get('category', 'general')
            confidence = question_data.get('confidence', 0.5)
            
            # Auto-classify question type and difficulty
            question_type = self.classify_question_type(question)
            difficulty = self.assess_difficulty_level(question, answer)
            
            # Extract keywords
            combined_text = f"{question} {answer}"
            keywords = self.extract_keywords(combined_text)
            
            # Calculate validation score
            validation_score = self.calculate_validation_score(question, answer, confidence)
            
            # Determine sub-category based on keywords and content
            sub_category = self.determine_sub_category(question, answer, category)
            
            # Create structured entry
            structured_entry = {
                "id": unique_id,
                "question": question,
                "answer": answer,
                "category": category,
                "sub_category": sub_category,
                "source_page": question_data.get('source_page', 0),
                "source_title": question_data.get('source_title', ''),
                "source_url": question_data.get('source_url', ''),
                "confidence": confidence,
                "question_type": question_type,
                "difficulty_level": difficulty,
                "keywords": keywords,
                "created_date": datetime.now().isoformat(),
                "modified_date": datetime.now().isoformat(),
                "review_status": "unreviewed",
                "reviewer_notes": "",
                "generation_method": question_data.get('type', 'automated'),
                "validation_score": validation_score
            }
            
            structured_questions.append(structured_entry)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} questions...")
        
        # Create final structured dataset
        structured_dataset = {
            "metadata": {
                "dataset_version": "1.0",
                "creation_date": datetime.now().isoformat(),
                "total_questions": len(structured_questions),
                "schema_version": "1.0",
                "categories": list(set(q['category'] for q in structured_questions)),
                "question_types": list(set(q['question_type'] for q in structured_questions)),
                "difficulty_levels": list(set(q['difficulty_level'] for q in structured_questions)),
                "average_validation_score": sum(q['validation_score'] for q in structured_questions) / len(structured_questions),
                "source_file": input_file
            },
            "schema": self.dataset_schema,
            "questions": structured_questions
        }
        
        # Save structured dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Structured dataset saved to: {output_file}")
        print(f"Total questions: {len(structured_questions)}")
        print(f"Average validation score: {structured_dataset['metadata']['average_validation_score']:.2f}")
        
        return structured_dataset
    
    def determine_sub_category(self, question: str, answer: str, category: str) -> str:
        """Determine sub-category based on content analysis"""
        sub_categories = {
            'admissions': ['fees', 'eligibility', 'exams', 'procedures', 'deadlines', 'documents'],
            'programs': ['undergraduate', 'postgraduate', 'specializations', 'curriculum', 'duration'],
            'facilities': ['hostels', 'library', 'labs', 'sports', 'dining', 'transport'],
            'rankings': ['national', 'international', 'accreditation', 'awards', 'recognition'],
            'faculty': ['qualifications', 'research', 'publications', 'experience'],
            'student_life': ['clubs', 'events', 'placements', 'internships', 'support'],
            'administration': ['governance', 'policies', 'leadership', 'contact'],
            'general': ['overview', 'history', 'location', 'campus']
        }
        
        combined_text = f"{question} {answer}".lower()
        category_subs = sub_categories.get(category, ['general'])
        
        for sub_cat in category_subs:
            if sub_cat in combined_text:
                return sub_cat
        
        return 'general'


class DatasetReviewGUI:
    """
    GUI application for reviewing and editing Q&A dataset
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KL University Q&A Dataset Review System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self.dataset = None
        self.current_index = 0
        self.total_questions = 0
        self.changes_made = False
        
        self.setup_gui()
        self.load_dataset()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Q&A Dataset Review System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Load/Save buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(button_frame, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Save Changes", command=self.save_dataset).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Export Review Report", command=self.export_review_report).pack(side=tk.LEFT, padx=(0, 10))
        
        # Progress frame
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="No dataset loaded")
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT)
        
        # Question navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀◀ First", command=self.first_question).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="◀ Previous", command=self.previous_question).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next ▶", command=self.next_question).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Last ▶▶", command=self.last_question).pack(side=tk.LEFT, padx=(0, 5))
        
        # Quick jump
        ttk.Label(nav_frame, text="Go to:").pack(side=tk.LEFT, padx=(20, 5))
        self.jump_var = tk.StringVar()
        jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_var, width=10)
        jump_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Jump", command=self.jump_to_question).pack(side=tk.LEFT)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Create notebook for organized view
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Question & Answer
        qa_frame = ttk.Frame(self.notebook)
        self.notebook.add(qa_frame, text="Question & Answer")
        
        # Question section
        ttk.Label(qa_frame, text="Question:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.question_text = tk.Text(qa_frame, height=3, width=80, wrap=tk.WORD, font=('Arial', 11))
        self.question_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Answer section
        ttk.Label(qa_frame, text="Answer:", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.answer_text = tk.Text(qa_frame, height=8, width=80, wrap=tk.WORD, font=('Arial', 11))
        self.answer_text.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        qa_frame.columnconfigure(0, weight=1)
        qa_frame.rowconfigure(3, weight=1)
        
        # Tab 2: Metadata
        metadata_frame = ttk.Frame(self.notebook)
        self.notebook.add(metadata_frame, text="Metadata")
        
        # Create metadata fields
        self.metadata_vars = {}
        metadata_fields = [
            ('ID', 'id', 'readonly'),
            ('Category', 'category', 'combo'),
            ('Sub Category', 'sub_category', 'entry'),
            ('Question Type', 'question_type', 'combo'),
            ('Difficulty Level', 'difficulty_level', 'combo'),
            ('Confidence', 'confidence', 'entry'),
            ('Validation Score', 'validation_score', 'readonly'),
            ('Source Page', 'source_page', 'entry'),
            ('Source Title', 'source_title', 'entry')
        ]
        
        for i, (label, field, widget_type) in enumerate(metadata_fields):
            ttk.Label(metadata_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            
            if widget_type == 'readonly':
                var = tk.StringVar()
                entry = ttk.Entry(metadata_frame, textvariable=var, state='readonly', width=50)
            elif widget_type == 'combo':
                var = tk.StringVar()
                if field == 'category':
                    values = ['admissions', 'programs', 'facilities', 'rankings', 'faculty', 'student_life', 'administration', 'general']
                elif field == 'question_type':
                    values = ['factual', 'procedural', 'comparative', 'numerical', 'yes_no', 'multiple_choice', 'descriptive', 'list_based']
                elif field == 'difficulty_level':
                    values = ['basic', 'intermediate', 'advanced', 'expert']
                entry = ttk.Combobox(metadata_frame, textvariable=var, values=values, width=47)
            else:
                var = tk.StringVar()
                entry = ttk.Entry(metadata_frame, textvariable=var, width=50)
            
            entry.grid(row=i, column=1, sticky=tk.W, pady=2)
            self.metadata_vars[field] = var
        
        # Tab 3: Review
        review_frame = ttk.Frame(self.notebook)
        self.notebook.add(review_frame, text="Review")
        
        # Review status
        ttk.Label(review_frame, text="Review Status:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.review_status_var = tk.StringVar()
        status_combo = ttk.Combobox(review_frame, textvariable=self.review_status_var, 
                                   values=['unreviewed', 'approved', 'needs_edit', 'rejected', 'flagged'],
                                   width=20)
        status_combo.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Keywords
        ttk.Label(review_frame, text="Keywords:", font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.keywords_text = tk.Text(review_frame, height=2, width=80, wrap=tk.WORD)
        self.keywords_text.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Reviewer notes
        ttk.Label(review_frame, text="Reviewer Notes:", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.notes_text = tk.Text(review_frame, height=6, width=80, wrap=tk.WORD)
        self.notes_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        review_frame.columnconfigure(0, weight=1)
        review_frame.rowconfigure(5, weight=1)
        
        # Action buttons at bottom
        action_frame = ttk.Frame(content_frame)
        action_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(action_frame, text="Mark as Approved", command=lambda: self.set_review_status('approved')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Mark as Needs Edit", command=lambda: self.set_review_status('needs_edit')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Mark as Rejected", command=lambda: self.set_review_status('rejected')).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(action_frame, text="Flag for Review", command=lambda: self.set_review_status('flagged')).pack(side=tk.LEFT, padx=(0, 10))
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(action_frame, text="Statistics", padding="5")
        stats_frame.pack(side=tk.RIGHT)
        
        self.stats_label = ttk.Label(stats_frame, text="No dataset loaded")
        self.stats_label.pack()
        
        # Bind events for auto-save
        self.question_text.bind('<KeyRelease>', self.on_text_change)
        self.answer_text.bind('<KeyRelease>', self.on_text_change)
        self.notes_text.bind('<KeyRelease>', self.on_text_change)
    
    def load_dataset(self):
        """Load structured dataset"""
        filename = filedialog.askopenfilename(
            title="Select Structured Dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="structured_qa_dataset.json"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
                
                self.current_index = 0
                self.total_questions = len(self.dataset.get('questions', []))
                self.dataset_filename = filename
                
                if self.total_questions > 0:
                    self.display_question()
                    self.update_progress()
                    self.update_statistics()
                    messagebox.showinfo("Success", f"Loaded {self.total_questions} questions")
                else:
                    messagebox.showwarning("Warning", "No questions found in dataset")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def display_question(self):
        """Display current question and metadata"""
        if not self.dataset or self.current_index >= self.total_questions:
            return
        
        question_data = self.dataset['questions'][self.current_index]
        
        # Clear and populate question/answer
        self.question_text.delete('1.0', tk.END)
        self.question_text.insert('1.0', question_data.get('question', ''))
        
        self.answer_text.delete('1.0', tk.END)
        self.answer_text.insert('1.0', question_data.get('answer', ''))
        
        # Populate metadata
        for field, var in self.metadata_vars.items():
            value = question_data.get(field, '')
            if isinstance(value, list):
                value = ', '.join(value)
            var.set(str(value))
        
        # Populate review fields
        self.review_status_var.set(question_data.get('review_status', 'unreviewed'))
        
        self.keywords_text.delete('1.0', tk.END)
        keywords = question_data.get('keywords', [])
        if isinstance(keywords, list):
            keywords = ', '.join(keywords)
        self.keywords_text.insert('1.0', keywords)
        
        self.notes_text.delete('1.0', tk.END)
        self.notes_text.insert('1.0', question_data.get('reviewer_notes', ''))
        
        self.update_progress()
    
    def save_current_question(self):
        """Save current question data"""
        if not self.dataset or self.current_index >= self.total_questions:
            return
        
        question_data = self.dataset['questions'][self.current_index]
        
        # Update question and answer
        question_data['question'] = self.question_text.get('1.0', tk.END).strip()
        question_data['answer'] = self.answer_text.get('1.0', tk.END).strip()
        
        # Update metadata
        for field, var in self.metadata_vars.items():
            if field not in ['id', 'validation_score']:  # Don't update readonly fields
                value = var.get()
                if field in ['confidence']:
                    try:
                        value = float(value)
                    except:
                        value = 0.5
                elif field in ['source_page']:
                    try:
                        value = int(value)
                    except:
                        value = 0
                question_data[field] = value
        
        # Update review fields
        question_data['review_status'] = self.review_status_var.get()
        question_data['reviewer_notes'] = self.notes_text.get('1.0', tk.END).strip()
        
        # Update keywords
        keywords_text = self.keywords_text.get('1.0', tk.END).strip()
        if keywords_text:
            question_data['keywords'] = [k.strip() for k in keywords_text.split(',')]
        
        # Update modified date
        question_data['modified_date'] = datetime.now().isoformat()
        
        self.changes_made = True
    
    def on_text_change(self, event=None):
        """Handle text changes"""
        self.changes_made = True
    
    def save_dataset(self):
        """Save the entire dataset"""
        if not self.dataset:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        # Save current question first
        self.save_current_question()
        
        # Update dataset metadata
        self.dataset['metadata']['last_modified'] = datetime.now().isoformat()
        
        try:
            # Save to original file
            with open(self.dataset_filename, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            
            # Also save a backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"dataset_backup_{timestamp}.json"
            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)
            
            self.changes_made = False
            messagebox.showinfo("Success", f"Dataset saved successfully!\nBackup created: {backup_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
    
    def export_review_report(self):
        """Export review statistics and report"""
        if not self.dataset:
            messagebox.showwarning("Warning", "No dataset loaded")
            return
        
        questions = self.dataset['questions']
        
        # Calculate statistics
        total_questions = len(questions)
        status_counts = {}
        category_counts = {}
        difficulty_counts = {}
        type_counts = {}
        
        approved_count = 0
        avg_confidence = 0
        avg_validation = 0
        
        for q in questions:
            status = q.get('review_status', 'unreviewed')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            category = q.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            difficulty = q.get('difficulty_level', 'unknown')
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            qtype = q.get('question_type', 'unknown')
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
            
            if status == 'approved':
                approved_count += 1
            
            avg_confidence += q.get('confidence', 0)
            avg_validation += q.get('validation_score', 0)
        
        avg_confidence /= total_questions
        avg_validation /= total_questions
        
        # Create report
        report = f"""
KL University Q&A Dataset Review Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
=========
Total Questions: {total_questions}
Approved Questions: {approved_count} ({100*approved_count/total_questions:.1f}%)
Average Confidence: {avg_confidence:.2f}
Average Validation Score: {avg_validation:.2f}

REVIEW STATUS DISTRIBUTION:
==========================
"""
        for status, count in sorted(status_counts.items()):
            percentage = 100 * count / total_questions
            report += f"{status.title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
CATEGORY DISTRIBUTION:
=====================
"""
        for category, count in sorted(category_counts.items()):
            percentage = 100 * count / total_questions
            report += f"{category.title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
DIFFICULTY DISTRIBUTION:
=======================
"""
        for difficulty, count in sorted(difficulty_counts.items()):
            percentage = 100 * count / total_questions
            report += f"{difficulty.title()}: {count} ({percentage:.1f}%)\n"
        
        report += f"""
QUESTION TYPE DISTRIBUTION:
==========================
"""
        for qtype, count in sorted(type_counts.items()):
            percentage = 100 * count / total_questions
            report += f"{qtype.title()}: {count} ({percentage:.1f}%)\n"
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"review_report_{timestamp}.txt"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Also create CSV for detailed analysis
            csv_data = []
            for q in questions:
                csv_data.append({
                    'id': q.get('id', ''),
                    'question': q.get('question', '')[:100] + '...' if len(q.get('question', '')) > 100 else q.get('question', ''),
                    'category': q.get('category', ''),
                    'sub_category': q.get('sub_category', ''),
                    'question_type': q.get('question_type', ''),
                    'difficulty_level': q.get('difficulty_level', ''),
                    'review_status': q.get('review_status', ''),
                    'confidence': q.get('confidence', 0),
                    'validation_score': q.get('validation_score', 0),
                    'source_page': q.get('source_page', 0),
                    'keywords': ', '.join(q.get('keywords', [])),
                    'reviewer_notes': q.get('reviewer_notes', '')
                })
            
            df = pd.DataFrame(csv_data)
            csv_filename = f"review_analysis_{timestamp}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            
            messagebox.showinfo("Success", f"Report exported:\n{report_filename}\n{csv_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
    
    def update_progress(self):
        """Update progress indicators"""
        if self.total_questions > 0:
            progress = (self.current_index + 1) / self.total_questions * 100
            self.progress_bar['value'] = progress
            self.progress_label.config(text=f"Question {self.current_index + 1} of {self.total_questions}")
        else:
            self.progress_bar['value'] = 0
            self.progress_label.config(text="No dataset loaded")
    
    def update_statistics(self):
        """Update statistics display"""
        if not self.dataset:
            self.stats_label.config(text="No dataset loaded")
            return
        
        questions = self.dataset['questions']
        approved = sum(1 for q in questions if q.get('review_status') == 'approved')
        needs_edit = sum(1 for q in questions if q.get('review_status') == 'needs_edit')
        rejected = sum(1 for q in questions if q.get('review_status') == 'rejected')
        
        stats_text = f"Approved: {approved} | Needs Edit: {needs_edit} | Rejected: {rejected}"
        self.stats_label.config(text=stats_text)
    
    def first_question(self):
        """Go to first question"""
        if self.dataset:
            self.save_current_question()
            self.current_index = 0
            self.display_question()
    
    def last_question(self):
        """Go to last question"""
        if self.dataset:
            self.save_current_question()
            self.current_index = self.total_questions - 1
            self.display_question()
    
    def previous_question(self):
        """Go to previous question"""
        if self.dataset and self.current_index > 0:
            self.save_current_question()
            self.current_index -= 1
            self.display_question()
    
    def next_question(self):
        """Go to next question"""
        if self.dataset and self.current_index < self.total_questions - 1:
            self.save_current_question()
            self.current_index += 1
            self.display_question()
    
    def jump_to_question(self):
        """Jump to specific question number"""
        try:
            question_num = int(self.jump_var.get())
            if 1 <= question_num <= self.total_questions:
                self.save_current_question()
                self.current_index = question_num - 1
                self.display_question()
            else:
                messagebox.showwarning("Warning", f"Question number must be between 1 and {self.total_questions}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid question number")
    
    def set_review_status(self, status):
        """Set review status for current question"""
        self.review_status_var.set(status)
        self.save_current_question()
        self.update_statistics()
        
        # Auto-advance to next unreviewed question if marking as approved/rejected
        if status in ['approved', 'rejected'] and self.current_index < self.total_questions - 1:
            self.next_question()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


def create_sample_structured_dataset():
    """Create a sample structured dataset for testing"""
    sample_questions = [
        {
            "question": "What is the application fee for B.Tech programs?",
            "answer": "The application fee is Rs. 1000/- and it is non-refundable.",
            "category": "admissions",
            "confidence": 0.9,
            "source_page": 3,
            "source_title": "Admissions - KL University",
            "type": "template_filled"
        },
        {
            "question": "What specializations are available in Computer Science Engineering?",
            "answer": "The Department of Computer Science & Engineering offers various specializations including Artificial Intelligence, Data Science, Cyber Security, and Cloud Computing.",
            "category": "programs", 
            "confidence": 0.8,
            "source_page": 16,
            "source_title": "Computer Science Engineering",
            "type": "template_direct"
        },
        {
            "question": "What is KL University's NIRF ranking?",
            "answer": "KL University secured 22nd rank in NIRF Rankings 2024 among all Universities in India.",
            "category": "rankings",
            "confidence": 0.95,
            "source_page": 2,
            "source_title": "Rankings - KL University",
            "type": "fact_based"
        }
    ]
    
    # Create sample raw dataset
    raw_dataset = {
        "questions": sample_questions,
        "metadata": {
            "total_questions": len(sample_questions),
            "generation_method": "sample_data"
        }
    }
    
    with open('sample_qa_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(raw_dataset, f, indent=2, ensure_ascii=False)
    
    print("✅ Sample dataset created: sample_qa_dataset.json")


def main():
    """Main execution function for Phase 3"""
    print("=== PHASE 3: Dataset Structure & GUI Review System ===")
    
    # Check if we have Phase 2 output
    input_files = ['kl_university_qa_dataset.json', 'sample_qa_dataset.json']
    input_file = None
    
    for file in input_files:
        try:
            with open(file, 'r') as f:
                input_file = file
                break
        except FileNotFoundError:
            continue
    
    if not input_file:
        print("No Phase 2 dataset found. Creating sample dataset...")
        create_sample_structured_dataset()
        input_file = 'sample_qa_dataset.json'
    
    # Step 1: Structure the dataset
    print(f"\nStep 1: Structuring dataset from {input_file}")
    structurer = DatasetStructurer()
    structured_dataset = structurer.structure_dataset(input_file, 'structured_qa_dataset.json')
    
    if structured_dataset:
        print(f"✅ Dataset structured successfully!")
        print(f"   - Total questions: {structured_dataset['metadata']['total_questions']}")
        print(f"   - Categories: {len(structured_dataset['metadata']['categories'])}")
        print(f"   - Average validation score: {structured_dataset['metadata']['average_validation_score']:.2f}")
        
        # Create additional exports
        print(f"\nStep 2: Creating additional exports...")
        
        # Export to CSV for analysis
        df_data = []
        for q in structured_dataset['questions']:
            df_data.append({
                'id': q['id'],
                'question': q['question'],
                'answer': q['answer'][:200] + '...' if len(q['answer']) > 200 else q['answer'],
                'category': q['category'],
                'sub_category': q['sub_category'],
                'question_type': q['question_type'],
                'difficulty_level': q['difficulty_level'],
                'confidence': q['confidence'],
                'validation_score': q['validation_score'],
                'review_status': q['review_status']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv('structured_qa_overview.csv', index=False, encoding='utf-8')
        print("✅ CSV overview created: structured_qa_overview.csv")
        
        # Create training-ready format
        training_format = {
            "dataset_info": {
                "name": "KL University Q&A Dataset",
                "version": "1.0",
                "description": "Structured Q&A dataset for KL University information",
                "total_questions": len(structured_dataset['questions'])
            },
            "data": []
        }
        
        for q in structured_dataset['questions']:
            training_format["data"].append({
                "id": q['id'],
                "input": q['question'],
                "output": q['answer'],
                "metadata": {
                    "category": q['category'],
                    "difficulty": q['difficulty_level'],
                    "confidence": q['confidence']
                }
            })
        
        with open('training_ready_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(training_format, f, indent=2, ensure_ascii=False)
        print("✅ Training-ready format created: training_ready_dataset.json")
        
        print(f"\nStep 3: Launching GUI Review System...")
        print("🚀 Starting GUI application for manual review...")
        
        # Launch GUI
        try:
            gui = DatasetReviewGUI()
            gui.run()
        except Exception as e:
            print(f"GUI Error: {e}")
            print("You can still use the structured dataset files created above.")
        
        print(f"\n=== PHASE 3 COMPLETE ===")
        print(f"Files created:")
        print(f"  📄 structured_qa_dataset.json - Main structured dataset")
        print(f"  📊 structured_qa_overview.csv - Dataset overview")
        print(f"  🎯 training_ready_dataset.json - Ready for ML training")
        print(f"  🔍 Use GUI to review and improve dataset quality")


def demo_structured_format():
    """Show example of structured format"""
    print("\n=== STRUCTURED DATASET FORMAT EXAMPLE ===")
    
    sample_entry = {
        "id": "qa_20241222_120000_abc12345",
        "question": "What is the application fee for B.Tech programs?",
        "answer": "The application fee is Rs. 1000/- and it is non-refundable.",
        "category": "admissions",
        "sub_category": "fees",
        "source_page": 3,
        "source_title": "Admissions - KL University",
        "source_url": "https://kluniversity.in/admissions-2025/",
        "confidence": 0.9,
        "question_type": "factual",
        "difficulty_level": "basic",
        "keywords": ["application", "fee", "btech", "programs"],
        "created_date": "2024-12-22T12:00:00",
        "modified_date": "2024-12-22T12:00:00",
        "review_status": "unreviewed",
        "reviewer_notes": "",
        "generation_method": "template_filled",
        "validation_score": 0.85
    }
    
    print(json.dumps(sample_entry, indent=2))
    print("\nThis structure provides:")
    print("✅ Unique identification for each Q&A pair")
    print("✅ Comprehensive metadata for analysis")
    print("✅ Quality metrics and validation scores")
    print("✅ Review workflow support")
    print("✅ Traceability to source content")


if __name__ == "__main__":
    # Show structured format example
    demo_structured_format()
    
    # Run main Phase 3 process
    main()