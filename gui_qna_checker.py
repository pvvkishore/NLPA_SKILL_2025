# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 07:49:52 2025

@author: Dr.PVVK
"""
"""
QA Dataset Verification and Refinement GUI
Interactive interface for reviewing and improving the generated QA dataset
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import os
from datetime import datetime

@dataclass
class QAPair:
    """Data class for Question-Answer pairs"""
    question: str
    answer: str
    context: str
    difficulty: str
    category: str
    source_page: str
    confidence: float = 0.0
    timestamp: str = ""
    verified: bool = False
    notes: str = ""

class QAVerificationGUI:
    """GUI application for QA dataset verification"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("QA Dataset Verification Tool")
        self.root.geometry("1200x800")
        
        # Data
        self.qa_pairs: List[QAPair] = []
        self.current_index = 0
        self.filtered_indices = []
        self.search_results = []
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.bind_shortcuts()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        style.configure('Status.TLabel', foreground='blue')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Success.TLabel', foreground='green')
        
    def create_widgets(self):
        """Create and layout GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Create sections
        self.create_toolbar(main_frame)
        self.create_navigation_section(main_frame)
        self.create_qa_display_section(main_frame)
        self.create_edit_section(main_frame)
        self.create_status_section(main_frame)
        
    def create_toolbar(self, parent):
        """Create toolbar with file operations"""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File operations
        ttk.Button(toolbar, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Save Dataset", command=self.save_dataset).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Export Verified", command=self.export_verified).pack(side=tk.LEFT, padx=(0, 5))
        
        # Separator
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # Statistics
        self.stats_label = ttk.Label(toolbar, text="No dataset loaded", style='Status.TLabel')
        self.stats_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Search
        ttk.Label(toolbar, text="Search:").pack(side=tk.RIGHT, padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=20)
        search_entry.pack(side=tk.RIGHT, padx=(0, 5))
        search_entry.bind('<Return>', self.search_qa_pairs)
        ttk.Button(toolbar, text="Search", command=self.search_qa_pairs).pack(side=tk.RIGHT)
        
    def create_navigation_section(self, parent):
        """Create navigation controls"""
        nav_frame = ttk.LabelFrame(parent, text="Navigation", padding="5")
        nav_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Navigation buttons
        button_frame = ttk.Frame(nav_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="First", command=self.first_qa).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Previous", command=self.previous_qa).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Next", command=self.next_qa).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Last", command=self.last_qa).pack(side=tk.LEFT, padx=(0, 5))
        
        # Position info
        self.position_label = ttk.Label(button_frame, text="0 / 0")
        self.position_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Jump to specific QA
        ttk.Label(button_frame, text="Go to:").pack(side=tk.RIGHT, padx=(0, 5))
        self.goto_var = tk.StringVar()
        goto_entry = ttk.Entry(button_frame, textvariable=self.goto_var, width=5)
        goto_entry.pack(side=tk.RIGHT, padx=(0, 5))
        goto_entry.bind('<Return>', self.goto_qa)
        
        # Filters
        filter_frame = ttk.Frame(nav_frame)
        filter_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, 
                                  values=["All", "Verified", "Unverified", "Easy", "Medium", "Hard"],
                                  state="readonly", width=10)
        filter_combo.pack(side=tk.LEFT, padx=(0, 5))
        filter_combo.bind('<<ComboboxSelected>>', self.apply_filter)
        
        # Category filter
        ttk.Label(filter_frame, text="Category:").pack(side=tk.LEFT, padx=(10, 5))
        self.category_filter_var = tk.StringVar(value="All")
        self.category_combo = ttk.Combobox(filter_frame, textvariable=self.category_filter_var, 
                                         state="readonly", width=15)
        self.category_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.category_combo.bind('<<ComboboxSelected>>', self.apply_filter)
        
    def create_qa_display_section(self, parent):
        """Create QA display section"""
        # Left side - QA Display
        display_frame = ttk.LabelFrame(parent, text="Question & Answer", padding="5")
        display_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        display_frame.rowconfigure(3, weight=1)
        
        # Question
        ttk.Label(display_frame, text="Question:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.question_text = scrolledtext.ScrolledText(display_frame, height=4, wrap=tk.WORD)
        self.question_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Answer
        ttk.Label(display_frame, text="Answer:", style='Header.TLabel').grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.answer_text = scrolledtext.ScrolledText(display_frame, height=6, wrap=tk.WORD)
        self.answer_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Context
        ttk.Label(display_frame, text="Context:", style='Header.TLabel').grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.context_text = scrolledtext.ScrolledText(display_frame, height=3, wrap=tk.WORD)
        self.context_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_edit_section(self, parent):
        """Create editing and verification section"""
        edit_frame = ttk.LabelFrame(parent, text="Edit & Verify", padding="5")
        edit_frame.grid(row=2, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        edit_frame.columnconfigure(1, weight=1)
        
        # Metadata
        row = 0
        ttk.Label(edit_frame, text="Difficulty:").grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        self.difficulty_var = tk.StringVar()
        difficulty_combo = ttk.Combobox(edit_frame, textvariable=self.difficulty_var, 
                                      values=["easy", "medium", "hard"], state="readonly", width=10)
        difficulty_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        
        row += 1
        ttk.Label(edit_frame, text="Category:").grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        self.category_var = tk.StringVar()
        self.category_edit_combo = ttk.Combobox(edit_frame, textvariable=self.category_var, width=15)
        self.category_edit_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        
        row += 1
        ttk.Label(edit_frame, text="Source:").grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        self.source_var = tk.StringVar()
        source_entry = ttk.Entry(edit_frame, textvariable=self.source_var)
        source_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        
        row += 1
        ttk.Label(edit_frame, text="Confidence:").grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        self.confidence_var = tk.DoubleVar()
        confidence_scale = ttk.Scale(edit_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        
        row += 1
        self.confidence_label = ttk.Label(edit_frame, text="0.0")
        self.confidence_label.grid(row=row, column=1, sticky=tk.W, pady=2)
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Verification status
        row += 1
        ttk.Separator(edit_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        row += 1
        self.verified_var = tk.BooleanVar()
        verified_check = ttk.Checkbutton(edit_frame, text="Verified", variable=self.verified_var)
        verified_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Notes
        row += 1
        ttk.Label(edit_frame, text="Notes:").grid(row=row, column=0, sticky=(tk.W, tk.N), padx=(0, 5))
        self.notes_text = scrolledtext.ScrolledText(edit_frame, height=4, width=30, wrap=tk.WORD)
        self.notes_text.grid(row=row, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=2)
        
        # Action buttons
        row += 1
        button_frame = ttk.Frame(edit_frame)
        button_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(button_frame, text="Save Changes", command=self.save_current_qa).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Delete QA", command=self.delete_current_qa).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Duplicate", command=self.duplicate_current_qa).pack(side=tk.LEFT)
        
        # Quick actions
        row += 1
        quick_frame = ttk.LabelFrame(edit_frame, text="Quick Actions", padding="5")
        quick_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(quick_frame, text="Mark Verified", command=lambda: self.quick_verify(True)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Mark Unverified", command=lambda: self.quick_verify(False)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Add Note", command=self.add_quick_note).pack(side=tk.LEFT)
        
    def create_status_section(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.load_dataset())
        self.root.bind('<Control-s>', lambda e: self.save_dataset())
        self.root.bind('<Control-Right>', lambda e: self.next_qa())
        self.root.bind('<Control-Left>', lambda e: self.previous_qa())
        self.root.bind('<Control-f>', lambda e: self.search_var.get() or self.search_qa_pairs())
        self.root.bind('<F1>', lambda e: self.show_help())
    
    # Dataset operations
    def load_dataset(self):
        """Load QA dataset from JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select QA Dataset",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.qa_pairs = []
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Standard format with metadata and qa_pairs
                qa_data = data.get('qa_pairs', [])
                if not qa_data and 'question' in data:
                    # Single QA pair format
                    qa_data = [data]
            elif isinstance(data, list):
                # Direct list of QA pairs
                qa_data = data
            else:
                raise ValueError("Invalid JSON format: expected dict or list")
            
            # Validate and load QA pairs
            if not qa_data:
                raise ValueError("No QA pairs found in the dataset")
            
            for i, qa in enumerate(qa_data):
                if not isinstance(qa, dict):
                    print(f"Warning: Skipping invalid QA pair at index {i} (not a dict)")
                    continue
                
                # Extract data with safe defaults
                try:
                    qa_pair = QAPair(
                        question=str(qa.get('question', '')),
                        answer=str(qa.get('answer', '')),
                        context=str(qa.get('context', '')),
                        difficulty=str(qa.get('difficulty', 'medium')),
                        category=str(qa.get('category', 'general')),
                        source_page=str(qa.get('source_page', '')),
                        confidence=float(qa.get('confidence', 0.0)),
                        timestamp=str(qa.get('timestamp', '')),
                        verified=bool(qa.get('verified', False)),
                        notes=str(qa.get('notes', ''))
                    )
                    
                    # Basic validation
                    if qa_pair.question.strip() and qa_pair.answer.strip():
                        self.qa_pairs.append(qa_pair)
                    else:
                        print(f"Warning: Skipping QA pair {i} - missing question or answer")
                        
                except (ValueError, TypeError) as e:
                    print(f"Warning: Skipping QA pair {i} - data conversion error: {e}")
                    continue
            
            if not self.qa_pairs:
                raise ValueError("No valid QA pairs could be loaded from the dataset")
            
            self.current_index = 0
            self.filtered_indices = list(range(len(self.qa_pairs)))
            self.update_category_filter()
            self.update_display()
            self.update_stats()
            self.set_status(f"Loaded {len(self.qa_pairs)} QA pairs from {os.path.basename(file_path)}")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON file: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            print(f"Debug info: {type(data) if 'data' in locals() else 'No data loaded'}")
            if 'data' in locals():
                print(f"Data structure: {str(data)[:200]}...")
    
    def save_dataset(self):
        """Save current QA dataset"""
        if not self.qa_pairs:
            messagebox.showwarning("Warning", "No dataset to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save QA Dataset",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Save current edits first
            self.save_current_qa_silent()
            
            dataset = {
                'metadata': {
                    'total_pairs': len(self.qa_pairs),
                    'verified_pairs': sum(1 for qa in self.qa_pairs if qa.verified),
                    'saved_at': datetime.now().isoformat(),
                    'source': 'QA Verification GUI'
                },
                'qa_pairs': [asdict(qa) for qa in self.qa_pairs]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            self.set_status(f"Saved dataset to {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save dataset: {str(e)}")
    
    def export_verified(self):
        """Export only verified QA pairs"""
        verified_pairs = [qa for qa in self.qa_pairs if qa.verified]
        
        if not verified_pairs:
            messagebox.showwarning("Warning", "No verified QA pairs to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Verified QA Pairs",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            dataset = {
                'metadata': {
                    'total_pairs': len(verified_pairs),
                    'exported_at': datetime.now().isoformat(),
                    'source': 'QA Verification GUI - Verified Only'
                },
                'qa_pairs': [asdict(qa) for qa in verified_pairs]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            self.set_status(f"Exported {len(verified_pairs)} verified QA pairs")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export verified pairs: {str(e)}")
    
    # Navigation methods
    def first_qa(self):
        """Go to first QA pair"""
        if self.filtered_indices:
            self.current_index = 0
            self.update_display()
    
    def last_qa(self):
        """Go to last QA pair"""
        if self.filtered_indices:
            self.current_index = len(self.filtered_indices) - 1
            self.update_display()
    
    def next_qa(self):
        """Go to next QA pair"""
        if self.filtered_indices and self.current_index < len(self.filtered_indices) - 1:
            self.save_current_qa_silent()
            self.current_index += 1
            self.update_display()
    
    def previous_qa(self):
        """Go to previous QA pair"""
        if self.filtered_indices and self.current_index > 0:
            self.save_current_qa_silent()
            self.current_index -= 1
            self.update_display()
    
    def goto_qa(self, event=None):
        """Go to specific QA pair by index"""
        try:
            target_index = int(self.goto_var.get()) - 1  # Convert to 0-based
            if 0 <= target_index < len(self.filtered_indices):
                self.save_current_qa_silent()
                self.current_index = target_index
                self.update_display()
            else:
                messagebox.showwarning("Warning", f"Index must be between 1 and {len(self.filtered_indices)}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid number")
        
        self.goto_var.set("")
    
    # Search and filter methods
    def search_qa_pairs(self, event=None):
        """Search QA pairs by question or answer content"""
        search_term = self.search_var.get().lower().strip()
        
        if not search_term:
            self.apply_filter()  # Reset to current filter
            return
        
        self.search_results = []
        for i, qa in enumerate(self.qa_pairs):
            if (search_term in qa.question.lower() or 
                search_term in qa.answer.lower() or 
                search_term in qa.category.lower() or
                search_term in qa.source_page.lower()):
                self.search_results.append(i)
        
        self.filtered_indices = self.search_results
        self.current_index = 0
        self.update_display()
        self.set_status(f"Found {len(self.search_results)} matches for '{search_term}'")
    
    def apply_filter(self, event=None):
        """Apply filters to QA pairs"""
        if not self.qa_pairs:
            return
        
        filter_type = self.filter_var.get()
        category_filter = self.category_filter_var.get()
        
        filtered = []
        for i, qa in enumerate(self.qa_pairs):
            # Apply main filter
            include = True
            if filter_type == "Verified" and not qa.verified:
                include = False
            elif filter_type == "Unverified" and qa.verified:
                include = False
            elif filter_type in ["Easy", "Medium", "Hard"] and qa.difficulty.lower() != filter_type.lower():
                include = False
            
            # Apply category filter
            if category_filter != "All" and qa.category != category_filter:
                include = False
            
            if include:
                filtered.append(i)
        
        self.filtered_indices = filtered
        self.current_index = 0
        self.update_display()
        self.set_status(f"Filtered to {len(filtered)} QA pairs")
    
    def update_category_filter(self):
        """Update category filter dropdown with available categories"""
        if not self.qa_pairs:
            return
        
        categories = set(qa.category for qa in self.qa_pairs)
        categories = ["All"] + sorted(list(categories))
        self.category_combo['values'] = categories
        self.category_edit_combo['values'] = sorted(list(categories - {"All"}))
    
    # Display methods
    def update_display(self):
        """Update the display with current QA pair"""
        if not self.filtered_indices:
            self.clear_display()
            return
        
        if self.current_index >= len(self.filtered_indices):
            self.current_index = 0
        
        actual_index = self.filtered_indices[self.current_index]
        qa = self.qa_pairs[actual_index]
        
        # Update text fields
        self.question_text.delete(1.0, tk.END)
        self.question_text.insert(1.0, qa.question)
        
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(1.0, qa.answer)
        
        self.context_text.delete(1.0, tk.END)
        self.context_text.insert(1.0, qa.context)
        
        # Update metadata
        self.difficulty_var.set(qa.difficulty)
        self.category_var.set(qa.category)
        self.source_var.set(qa.source_page)
        self.confidence_var.set(qa.confidence)
        self.verified_var.set(qa.verified)
        
        # Update notes
        self.notes_text.delete(1.0, tk.END)
        self.notes_text.insert(1.0, qa.notes)
        
        # Update position
        self.position_label.config(text=f"{self.current_index + 1} / {len(self.filtered_indices)}")
        
        # Update progress
        if len(self.filtered_indices) > 0:
            progress = (self.current_index + 1) / len(self.filtered_indices) * 100
            self.progress_var.set(progress)
    
    def clear_display(self):
        """Clear all display fields"""
        self.question_text.delete(1.0, tk.END)
        self.answer_text.delete(1.0, tk.END)
        self.context_text.delete(1.0, tk.END)
        self.notes_text.delete(1.0, tk.END)
        
        self.difficulty_var.set("")
        self.category_var.set("")
        self.source_var.set("")
        self.confidence_var.set(0.0)
        self.verified_var.set(False)
        
        self.position_label.config(text="0 / 0")
        self.progress_var.set(0)
    
    def update_confidence_label(self, value):
        """Update confidence label with current value"""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    # Edit methods
    def save_current_qa(self):
        """Save changes to current QA pair"""
        if not self.filtered_indices:
            return
        
        actual_index = self.filtered_indices[self.current_index]
        qa = self.qa_pairs[actual_index]
        
        # Get values from GUI
        qa.question = self.question_text.get(1.0, tk.END).strip()
        qa.answer = self.answer_text.get(1.0, tk.END).strip()
        qa.context = self.context_text.get(1.0, tk.END).strip()
        qa.difficulty = self.difficulty_var.get()
        qa.category = self.category_var.get()
        qa.source_page = self.source_var.get()
        qa.confidence = self.confidence_var.get()
        qa.verified = self.verified_var.get()
        qa.notes = self.notes_text.get(1.0, tk.END).strip()
        qa.timestamp = datetime.now().isoformat()
        
        self.update_category_filter()
        self.update_stats()
        self.set_status("Changes saved")
    
    def save_current_qa_silent(self):
        """Save current QA without status message"""
        if not self.filtered_indices:
            return
        
        actual_index = self.filtered_indices[self.current_index]
        qa = self.qa_pairs[actual_index]
        
        qa.question = self.question_text.get(1.0, tk.END).strip()
        qa.answer = self.answer_text.get(1.0, tk.END).strip()
        qa.context = self.context_text.get(1.0, tk.END).strip()
        qa.difficulty = self.difficulty_var.get()
        qa.category = self.category_var.get()
        qa.source_page = self.source_var.get()
        qa.confidence = self.confidence_var.get()
        qa.verified = self.verified_var.get()
        qa.notes = self.notes_text.get(1.0, tk.END).strip()
    
    def delete_current_qa(self):
        """Delete current QA pair"""
        if not self.filtered_indices:
            return
        
        if messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this QA pair?"):
            actual_index = self.filtered_indices[self.current_index]
            del self.qa_pairs[actual_index]
            
            # Update filtered indices
            self.filtered_indices = [i if i < actual_index else i-1 for i in self.filtered_indices if i != actual_index]
            
            # Adjust current index
            if self.current_index >= len(self.filtered_indices):
                self.current_index = len(self.filtered_indices) - 1
            
            self.update_display()
            self.update_stats()
            self.set_status("QA pair deleted")
    
    def duplicate_current_qa(self):
        """Create a duplicate of current QA pair"""
        if not self.filtered_indices:
            return
        
        actual_index = self.filtered_indices[self.current_index]
        original_qa = self.qa_pairs[actual_index]
        
        # Create duplicate
        duplicate_qa = QAPair(
            question=original_qa.question + " (Copy)",
            answer=original_qa.answer,
            context=original_qa.context,
            difficulty=original_qa.difficulty,
            category=original_qa.category,
            source_page=original_qa.source_page,
            confidence=original_qa.confidence,
            verified=False,
            notes="Duplicated from another QA pair",
            timestamp=datetime.now().isoformat()
        )
        
        # Insert after current
        self.qa_pairs.insert(actual_index + 1, duplicate_qa)
        
        # Update filtered indices
        self.filtered_indices = [i if i <= actual_index else i+1 for i in self.filtered_indices]
        self.filtered_indices.insert(self.current_index + 1, actual_index + 1)
        
        # Move to duplicate
        self.current_index += 1
        self.update_display()
        self.update_stats()
        self.set_status("QA pair duplicated")
    
    # Quick actions
    def quick_verify(self, verified):
        """Quickly mark as verified/unverified"""
        self.verified_var.set(verified)
        self.save_current_qa()
        if verified:
            self.next_qa()  # Auto-advance after verification
    
    def add_quick_note(self):
        """Add a quick note"""
        note = tk.simpledialog.askstring("Add Note", "Enter note:")
        if note:
            current_notes = self.notes_text.get(1.0, tk.END).strip()
            if current_notes:
                current_notes += "\n"
            current_notes += f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}"
            
            self.notes_text.delete(1.0, tk.END)
            self.notes_text.insert(1.0, current_notes)
            self.save_current_qa()
    
    # Status and statistics
    def update_stats(self):
        """Update statistics display"""
        if not self.qa_pairs:
            self.stats_label.config(text="No dataset loaded")
            return
        
        total = len(self.qa_pairs)
        verified = sum(1 for qa in self.qa_pairs if qa.verified)
        percentage = (verified / total * 100) if total > 0 else 0
        
        self.stats_label.config(text=f"Total: {total} | Verified: {verified} ({percentage:.1f}%)")
    
    def set_status(self, message):
        """Set status bar message"""
        self.status_label.config(text=message)
        self.root.after(3000, lambda: self.status_label.config(text="Ready"))
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
QA Dataset Verification Tool - Help

Keyboard Shortcuts:
• Ctrl+O: Load dataset
• Ctrl+S: Save dataset
• Ctrl+→: Next QA pair
• Ctrl+←: Previous QA pair
• Ctrl+F: Focus search
• F1: Show this help

Navigation:
• Use First/Previous/Next/Last buttons
• Or enter a number in "Go to" field
• Use filters to show specific subsets

Editing:
• Modify question, answer, or context directly
• Change metadata (difficulty, category, etc.)
• Mark as verified when satisfied
• Add notes for future reference

Quick Actions:
• Mark Verified: Sets verified flag and advances
• Mark Unverified: Removes verified flag
• Add Note: Quickly add timestamped note

Tips:
• Changes are auto-saved when navigating
• Use filters to focus on unverified items
• Export verified pairs for final dataset
• Search works across all text fields
        """
        
        messagebox.showinfo("Help", help_text)

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = QAVerificationGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()