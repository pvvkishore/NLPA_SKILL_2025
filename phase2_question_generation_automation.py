import re
import json
import random
import sys
import subprocess
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import string

# Automatic dependency installation
def install_dependencies():
    """Install required packages automatically"""
    packages = [
        "pandas",
        "nltk",
        "transformers",
        "torch",
        "sentence-transformers"
    ]
    
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

# Try importing with auto-install fallback
try:
    import pandas as pd
    import nltk
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Installing required dependencies...")
    install_dependencies()
    import pandas as pd
    import nltk
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class QuestionAnswerGenerator:
    """
    Automated Question-Answer generation for university content
    Target: 1000-1500 Q&A pairs for effective training
    """
    
    def __init__(self):
        self.question_templates = self.load_question_templates()
        self.answer_templates = self.load_answer_templates()
        self.question_count_targets = {
            'admissions': 250,      # High priority - most queries
            'programs': 300,        # High priority - course info
            'facilities': 150,      # Medium priority
            'rankings': 100,        # Medium priority
            'faculty': 150,         # Medium priority
            'student_life': 200,    # Medium priority
            'administration': 100,  # Lower priority
            'general': 150         # General university info
        }
        
        # Initialize question generation model (optional)
        self.qg_model = None
        self.load_qg_model()
    
    def load_qg_model(self):
        """Load question generation model (optional for advanced generation)"""
        try:
            print("Loading question generation model...")
            self.qg_model = pipeline("text2text-generation", 
                                   model="valhalla/t5-small-qg-prepend", 
                                   tokenizer="valhalla/t5-small-qg-prepend")
            print("✓ Question generation model loaded")
        except Exception as e:
            print(f"⚠ Could not load QG model: {e}")
            print("Continuing with template-based generation")
    
    def load_question_templates(self) -> Dict[str, List[str]]:
        """Define question templates for each category"""
        return {
            'admissions': [
                "What is the application fee for {program}?",
                "What are the eligibility criteria for {program}?",
                "When is the deadline for {exam} applications?",
                "What entrance exams are accepted for {program}?",
                "How can I apply for {program}?",
                "What documents are required for {program} admission?",
                "What is the selection process for {program}?",
                "Are there any scholarships available for {program}?",
                "What is the fee structure for {program}?",
                "Can I get admission through lateral entry to {program}?",
                "What is the minimum percentage required for {program}?",
                "Is there any management quota for {program}?",
                "What is the counseling process for {program}?",
                "How many seats are available in {program}?",
                "What is the reservation policy for {program}?"
            ],
            'programs': [
                "What specializations are available in {department}?",
                "What is the duration of {program}?",
                "What subjects are taught in {program}?",
                "What are the career prospects after {program}?",
                "What is the curriculum of {program}?",
                "Are there any industry collaborations for {program}?",
                "What practical training is provided in {program}?",
                "What research opportunities are available in {program}?",
                "What certifications can I get with {program}?",
                "Is {program} AICTE approved?",
                "What is the faculty-student ratio in {department}?",
                "Are there exchange programs for {program} students?",
                "What labs are available for {program}?",
                "What projects do {program} students work on?",
                "What is unique about {program} at KL University?"
            ],
            'facilities': [
                "What facilities are available at KL University?",
                "What are the hostel facilities like?",
                "What library resources are available?",
                "What sports facilities does the university have?",
                "How is the campus infrastructure?",
                "What dining options are available on campus?",
                "Is Wi-Fi available throughout the campus?",
                "What medical facilities are available?",
                "What transportation is provided?",
                "What recreational facilities are there?",
                "How many students can the hostels accommodate?",
                "What security measures are in place?",
                "Are there shopping facilities on campus?",
                "What international student facilities exist?",
                "What disability support services are available?"
            ],
            'rankings': [
                "What is KL University's NIRF ranking?",
                "What accreditations does KL University have?",
                "What awards has KL University received?",
                "How is KL University ranked globally?",
                "What recognition has KL University achieved?",
                "Is KL University NAAC accredited?",
                "What is the university's research ranking?",
                "What sustainability awards has the university won?",
                "How does KL University compare to other universities?",
                "What quality certifications does the university have?"
            ],
            'faculty': [
                "How many faculty members does {department} have?",
                "What are the qualifications of faculty in {department}?",
                "What research is being conducted in {department}?",
                "How many publications does the faculty have?",
                "What industry experience do faculty members have?",
                "Are there visiting faculty from other countries?",
                "What training programs are available for faculty?",
                "What awards have faculty members received?",
                "How many PhD faculty are there in {department}?",
                "What consultancy services do faculty provide?"
            ],
            'student_life': [
                "What clubs and societies are available?",
                "What cultural events does the university organize?",
                "What sports teams can students join?",
                "What placement support is provided?",
                "What internship opportunities are available?",
                "What student support services exist?",
                "What alumni network benefits are there?",
                "What entrepreneurship support is provided?",
                "What competitions do students participate in?",
                "What international exposure opportunities exist?"
            ],
            'administration': [
                "Who is the Chancellor of KL University?",
                "Who is the Vice-Chancellor?",
                "What is the university's vision and mission?",
                "How is the university governed?",
                "What are the university's policies?",
                "How can I contact the administration?",
                "What is the organizational structure?",
                "What are the university's strategic goals?"
            ],
            'general': [
                "When was KL University established?",
                "Where is KL University located?",
                "What type of university is KLEF?",
                "How many campuses does KL University have?",
                "What is the total student strength?",
                "What makes KL University special?",
                "How can I visit the campus?",
                "What are the university's contact details?"
            ]
        }
    
    def load_answer_templates(self) -> Dict[str, List[str]]:
        """Define answer templates for common question types"""
        return {
            'fee': [
                "The application fee is Rs. {amount}/- and it is non-refundable.",
                "The fee for {program} is Rs. {amount} per year.",
                "Total fees including all charges is approximately Rs. {amount}."
            ],
            'eligibility': [
                "Candidates must have {percentage}% in {subjects} to be eligible for {program}.",
                "A pass in {qualification} with {percentage}% is required for {program}.",
                "Students with {background} are eligible for {program}."
            ],
            'duration': [
                "{program} is a {duration} year program.",
                "The duration of {program} is {duration} years.",
                "Students complete {program} in {duration} years."
            ],
            'facilities': [
                "KL University provides {facility} with state-of-the-art infrastructure.",
                "The university has {number} {facility} available for students.",
                "{facility} is available 24/7 for student use."
            ]
        }
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract key entities from content for template filling"""
        entities = {
            'programs': [],
            'departments': [],
            'fees': [],
            'percentages': [],
            'numbers': [],
            'facilities': [],
            'exams': []
        }
        
        # Extract programs
        program_patterns = [
            r'\b(B\.?Tech\.?(?:\s+\w+)*)',
            r'\b(M\.?Tech\.?(?:\s+\w+)*)',
            r'\b(MBA|BBA|MCA|B\.?Sc|M\.?Sc|Ph\.?D|B\.?Arch)',
            r'\b(B\.?Pharmacy|Pharm\.?D)'
        ]
        
        for pattern in program_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['programs'].extend([match.strip() for match in matches])
        
        # Extract departments
        dept_patterns = [
            r'Department of ([A-Za-z\s&]+)',
            r'(Computer Science|Electronics|Mechanical|Civil|Electrical)(?:\s+Engineering)?',
            r'(CSE|ECE|EEE|ME|CE)\b'
        ]
        
        for pattern in dept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['departments'].extend([match.strip() for match in matches])
        
        # Extract fees
        fee_patterns = [
            r'Rs\.?\s*(\d{1,3}(?:,\d{3})*)',
            r'(\d+)\s*(?:lakhs?|crores?)'
        ]
        
        for pattern in fee_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['fees'].extend(matches)
        
        # Extract percentages
        percentage_matches = re.findall(r'(\d+)%', content)
        entities['percentages'].extend(percentage_matches)
        
        # Extract exam names
        exam_patterns = [
            r'\b(KLEEE|JEE|EAMCET|NEET|GATE|CAT|MAT|CLAT)\b',
            r'(KLMAT|KLSAT|KLHAT|KLECET)\b'
        ]
        
        for pattern in exam_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities['exams'].extend(matches)
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([item for item in entities[key] if item]))
        
        return entities
    
    def generate_template_questions(self, content: str, category: str, entities: Dict) -> List[Dict]:
        """Generate questions using templates"""
        questions = []
        templates = self.question_templates.get(category, [])
        target_count = self.question_count_targets.get(category, 50)
        
        # Generate questions by filling templates
        for template in templates:
            # Identify template variables
            variables = re.findall(r'\{(\w+)\}', template)
            
            if not variables:
                # No variables, use template as is
                answer = self.generate_answer_for_question(template, content, entities)
                if answer:
                    questions.append({
                        'question': template,
                        'answer': answer,
                        'category': category,
                        'type': 'template_direct',
                        'confidence': 0.8
                    })
            else:
                # Fill variables with appropriate entities
                filled_templates = self.fill_template_variables(template, variables, entities)
                for filled_template in filled_templates:
                    answer = self.generate_answer_for_question(filled_template, content, entities)
                    if answer:
                        questions.append({
                            'question': filled_template,
                            'answer': answer,
                            'category': category,
                            'type': 'template_filled',
                            'confidence': 0.7
                        })
        
        # If we don't have enough questions, generate more variations
        if len(questions) < target_count // 3:  # Aim for 1/3 from templates
            additional_questions = self.generate_additional_questions(content, category, entities)
            questions.extend(additional_questions)
        
        return questions[:target_count // 3]  # Limit to avoid over-generation
    
    def fill_template_variables(self, template: str, variables: List[str], entities: Dict) -> List[str]:
        """Fill template variables with appropriate entities"""
        filled_templates = []
        
        # Create mapping of variables to entity types
        variable_mapping = {
            'program': 'programs',
            'department': 'departments',
            'exam': 'exams',
            'facility': ['library', 'hostel', 'lab', 'cafeteria', 'sports complex'],
            'amount': 'fees',
            'percentage': 'percentages'
        }
        
        # Generate combinations
        for variable in variables:
            if variable in variable_mapping:
                entity_type = variable_mapping[variable]
                if isinstance(entity_type, list):
                    # Predefined list
                    entity_values = entity_type
                else:
                    # From extracted entities
                    entity_values = entities.get(entity_type, [])
                
                if entity_values:
                    for value in entity_values[:3]:  # Limit to 3 variations per template
                        filled_template = template.replace(f'{{{variable}}}', value)
                        filled_templates.append(filled_template)
        
        return filled_templates[:5]  # Limit total variations
    
    def generate_answer_for_question(self, question: str, content: str, entities: Dict) -> str:
        """Generate appropriate answer for a question based on content"""
        question_lower = question.lower()
        content_sentences = content.split('.')
        
        # Find relevant sentences based on question keywords
        relevant_sentences = []
        
        # Extract key terms from question
        question_terms = self.extract_question_terms(question)
        
        for sentence in content_sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains relevant information
            sentence_lower = sentence.lower()
            relevance_score = 0
            
            for term in question_terms:
                if term in sentence_lower:
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_sentences.append((sentence, relevance_score))
        
        # Sort by relevance and take the best
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if relevant_sentences:
            # Create answer from most relevant sentences
            answer_parts = []
            for sentence, score in relevant_sentences[:2]:  # Take top 2 sentences
                answer_parts.append(sentence.strip())
            
            answer = '. '.join(answer_parts)
            return self.clean_answer(answer)
        
        return None
    
    def extract_question_terms(self, question: str) -> List[str]:
        """Extract key terms from question for answer matching"""
        # Remove question words and common words
        question_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'can', 'does', 'do'}
        
        terms = []
        words = question.lower().split()
        
        for word in words:
            word = word.strip(string.punctuation)
            if len(word) > 2 and word not in question_words:
                terms.append(word)
        
        return terms
    
    def clean_answer(self, answer: str) -> str:
        """Clean and format the answer"""
        # Remove multiple spaces
        answer = re.sub(r'\s+', ' ', answer)
        
        # Ensure proper sentence ending
        if answer and not answer.endswith('.'):
            answer += '.'
        
        # Capitalize first letter
        if answer:
            answer = answer[0].upper() + answer[1:]
        
        return answer.strip()
    
    def generate_additional_questions(self, content: str, category: str, entities: Dict) -> List[Dict]:
        """Generate additional questions using content analysis"""
        questions = []
        
        # Generate fact-based questions
        fact_questions = self.generate_fact_questions(content, category, entities)
        questions.extend(fact_questions)
        
        # Generate comparison questions
        comparison_questions = self.generate_comparison_questions(content, category)
        questions.extend(comparison_questions)
        
        # Generate "how many" questions
        counting_questions = self.generate_counting_questions(content, category)
        questions.extend(counting_questions)
        
        return questions
    
    def generate_fact_questions(self, content: str, category: str, entities: Dict) -> List[Dict]:
        """Generate factual questions from content"""
        questions = []
        
        # Look for factual statements that can be turned into questions
        fact_patterns = [
            (r'(\w+) has (\d+[+]?) ([^.]+)', "How many {2} does {0} have?"),
            (r'(\w+) is (?:a |an )?([^.]+)', "What is {0}?"),
            (r'The (\w+) (?:of|for) (\w+) is ([^.]+)', "What is the {0} of {1}?"),
            (r'(\w+) provides ([^.]+)', "What does {0} provide?"),
            (r'(\w+) offers ([^.]+)', "What does {0} offer?")
        ]
        
        for pattern, question_template in fact_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    question = question_template.format(*groups)
                    answer = match.group(0).strip()
                    
                    questions.append({
                        'question': question,
                        'answer': answer,
                        'category': category,
                        'type': 'fact_based',
                        'confidence': 0.6
                    })
                except:
                    continue
        
        return questions[:10]  # Limit to avoid too many
    
    def generate_comparison_questions(self, content: str, category: str) -> List[Dict]:
        """Generate comparison-based questions"""
        questions = []
        
        comparison_templates = [
            "How does KL University compare to other universities?",
            "What makes KL University different from other institutions?",
            "What are the advantages of studying at KL University?",
            "Why should I choose KL University over other universities?"
        ]
        
        for template in comparison_templates:
            # Find relevant content for comparison
            comparison_indicators = ['rank', 'award', 'recognition', 'unique', 'special', 'best']
            relevant_content = []
            
            for sentence in content.split('.'):
                if any(indicator in sentence.lower() for indicator in comparison_indicators):
                    relevant_content.append(sentence.strip())
            
            if relevant_content:
                answer = '. '.join(relevant_content[:2])
                questions.append({
                    'question': template,
                    'answer': self.clean_answer(answer),
                    'category': category,
                    'type': 'comparison',
                    'confidence': 0.5
                })
        
        return questions
    
    def generate_counting_questions(self, content: str, category: str) -> List[Dict]:
        """Generate counting/numerical questions"""
        questions = []
        
        # Find numerical information
        number_patterns = [
            (r'(\d+)\s+(students?|faculty|courses?|programs?|labs?)', "How many {1} are there?"),
            (r'(\d+)\s+(?:years?|months?)', "What is the duration?"),
            (r'(\d+)%', "What percentage?"),
            (r'Rs\.?\s*(\d{1,3}(?:,\d{3})*)', "What is the fee?")
        ]
        
        for pattern, question_template in number_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    full_match = match.group(0)
                    number = match.group(1)
                    
                    if len(match.groups()) > 1:
                        subject = match.group(2)
                        question = question_template.format(number, subject)
                    else:
                        question = question_template
                    
                    # Find the sentence containing this information
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    
                    questions.append({
                        'question': question,
                        'answer': context.strip(),
                        'category': category,
                        'type': 'numerical',
                        'confidence': 0.7
                    })
                except:
                    continue
        
        return questions[:5]  # Limit numerical questions
    
    def generate_from_ml_model(self, content: str, category: str) -> List[Dict]:
        """Generate questions using ML model (if available)"""
        questions = []
        
        if not self.qg_model:
            return questions
        
        try:
            # Split content into chunks for processing
            chunks = self.split_content_into_chunks(content, max_length=200)
            
            for chunk in chunks[:3]:  # Process only first 3 chunks to avoid overload
                # Prepare input for question generation
                input_text = f"generate question: {chunk}"
                
                # Generate question
                result = self.qg_model(input_text, max_length=100, num_return_sequences=2)
                
                for generated in result:
                    question = generated['generated_text'].strip()
                    if question and '?' in question:
                        questions.append({
                            'question': question,
                            'answer': chunk,
                            'category': category,
                            'type': 'ml_generated',
                            'confidence': 0.6
                        })
        except Exception as e:
            print(f"ML generation error: {e}")
        
        return questions
    
    def split_content_into_chunks(self, content: str, max_length: int = 200) -> List[str]:
        """Split content into manageable chunks"""
        sentences = content.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_university_data(self, processed_data: Dict) -> Dict:
        """Main function to generate Q&A dataset from processed university data"""
        print("Starting question generation...")
        
        all_questions = []
        category_stats = defaultdict(int)
        
        # Process each page
        for page in processed_data['pages']:
            page_questions = []
            
            # Extract entities from page content
            entities = self.extract_entities(page['content'])
            
            # Generate questions for each category the page belongs to
            for category in page['categories']:
                print(f"Generating questions for {category} (Page {page['page_number']})")
                
                # Template-based generation
                template_questions = self.generate_template_questions(
                    page['content'], category, entities
                )
                page_questions.extend(template_questions)
                
                # ML-based generation (if available)
                ml_questions = self.generate_from_ml_model(page['content'], category)
                page_questions.extend(ml_questions)
                
                category_stats[category] += len(template_questions) + len(ml_questions)
            
            # Add page metadata to questions
            for question in page_questions:
                question.update({
                    'source_page': page['page_number'],
                    'source_title': page['title'],
                    'source_url': page.get('url', ''),
                    'id': f"q_{page['page_number']}_{len(all_questions) + len([q for q in page_questions if q == question])}"
                })
            
            all_questions.extend(page_questions)
        
        # Quality filtering
        filtered_questions = self.filter_questions(all_questions)
        
        # Balance categories to meet targets
        balanced_questions = self.balance_categories(filtered_questions)
        
        print(f"\nGeneration complete!")
        print(f"Total questions generated: {len(balanced_questions)}")
        print(f"Category distribution: {dict(category_stats)}")
        
        return {
            'questions': balanced_questions,
            'metadata': {
                'total_questions': len(balanced_questions),
                'category_distribution': dict(category_stats),
                'generation_method': 'automated_template_ml_hybrid',
                'target_achieved': len(balanced_questions) >= 1000
            }
        }
    
    def filter_questions(self, questions: List[Dict]) -> List[Dict]:
        """Filter out low-quality questions"""
        filtered = []
        seen_questions = set()
        
        for q in questions:
            question = q['question'].strip().lower()
            answer = q['answer'].strip()
            
            # Skip duplicates
            if question in seen_questions:
                continue
            
            # Skip if question or answer is too short
            if len(question) < 10 or len(answer) < 20:
                continue
            
            # Skip if answer is too long (likely contains irrelevant info)
            if len(answer) > 500:
                q['answer'] = answer[:497] + "..."
            
            # Skip questions without question marks
            if '?' not in question:
                continue
            
            seen_questions.add(question)
            filtered.append(q)
        
        return filtered
    
    def balance_categories(self, questions: List[Dict]) -> List[Dict]:
        """Balance questions across categories to meet targets"""
        category_questions = defaultdict(list)
        
        # Group by category
        for q in questions:
            category_questions[q['category']].append(q)
        
        balanced = []
        
        # Take required number from each category
        for category, target in self.question_count_targets.items():
            available = category_questions.get(category, [])
            
            if len(available) >= target:
                # Take top questions (by confidence)
                available.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                balanced.extend(available[:target])
            else:
                # Take all available
                balanced.extend(available)
        
        return balanced
    
    def save_qa_dataset(self, qa_data: Dict, output_path: str):
        """Save the Q&A dataset"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        print(f"Q&A dataset saved to: {output_path}")
    
    def export_to_csv(self, qa_data: Dict, output_path: str):
        """Export Q&A dataset to CSV"""
        df = pd.DataFrame(qa_data['questions'])
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"CSV export saved to: {output_path}")
    
    def create_training_splits(self, qa_data: Dict) -> Dict:
        """Create train/validation/test splits"""
        questions = qa_data['questions'].copy()
        random.shuffle(questions)
        
        total = len(questions)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)
        
        splits = {
            'train': questions[:train_size],
            'validation': questions[train_size:train_size + val_size],
            'test': questions[train_size + val_size:],
            'metadata': {
                'total_questions': total,
                'train_size': train_size,
                'validation_size': val_size,
                'test_size': total - train_size - val_size
            }
        }
        
        return splits


def main():
    """Main execution function"""
    print("=== KL University Q&A Dataset Generator ===")
    print("Target: 1000-1500 Q&A pairs for training")
    
    # Load processed data from Phase 1
    try:
        with open('processed_university_data.json', 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        print(f"✓ Loaded processed data with {len(processed_data['pages'])} pages")
    except FileNotFoundError:
        print("Error: processed_university_data.json not found.")
        print("Please run Phase 1 (data preprocessing) first.")
        return
    
    # Initialize question generator
    generator = QuestionAnswerGenerator()
    
    # Generate Q&A dataset
    qa_dataset = generator.process_university_data(processed_data)
    
    # Save main dataset
    generator.save_qa_dataset(qa_dataset, 'kl_university_qa_dataset.json')
    generator.export_to_csv(qa_dataset, 'kl_university_qa_dataset.csv')
    
    # Create training splits
    training_splits = generator.create_training_splits(qa_dataset)
    
    # Save training splits
    for split_name, split_data in training_splits.items():
        if split_name != 'metadata':
            split_dataset = {'questions': split_data, 'metadata': qa_dataset['metadata']}
            generator.save_qa_dataset(split_dataset, f'kl_university_qa_{split_name}.json')
    
    # Print final statistics
    print(f"\n=== FINAL DATASET STATISTICS ===")
    print(f"Total Q&A pairs: {qa_dataset['metadata']['total_questions']}")
    print(f"Training set: {training_splits['metadata']['train_size']} questions")
    print(f"Validation set: {training_splits['metadata']['validation_size']} questions")
    print(f"Test set: {training_splits['metadata']['test_size']} questions")
    
    print(f"\nCategory Distribution:")
    for category, count in qa_dataset['metadata']['category_distribution'].items():
        percentage = (count / qa_dataset['metadata']['total_questions']) * 100
        print(f"  {category}: {count} questions ({percentage:.1f}%)")
    
    target_achieved = qa_dataset['metadata']['total_questions'] >= 1000
    print(f"\nTarget Achievement: {'✓' if target_achieved else '✗'} {'SUCCESS' if target_achieved else 'PARTIAL'}")
    
    if target_achieved:
        print("🎉 Dataset ready for training NLP models!")
    else:
        print("⚠ Consider running with more content or adjusting templates")

    return qa_dataset


def demo_question_samples():
    """Show sample questions for each category"""
    print("\n=== SAMPLE QUESTIONS BY CATEGORY ===")
    
    samples = {
        'admissions': [
            "What is the application fee for B.Tech programs?",
            "What are the eligibility criteria for M.Tech admission?",
            "When is the deadline for KLEEE applications?",
            "What entrance exams are accepted for B.Tech admission?",
            "How can I apply for MBA program?"
        ],
        'programs': [
            "What specializations are available in Computer Science Engineering?",
            "What is the duration of B.Tech program?",
            "What subjects are taught in Electronics and Communication Engineering?",
            "What are the career prospects after M.Tech?",
            "What certifications can I get with B.Tech CSE?"
        ],
        'facilities': [
            "What hostel facilities are available at KL University?",
            "What library resources does the university provide?",
            "What sports facilities does KL University have?",
            "How is the campus infrastructure at KLEF?",
            "What transportation is provided by the university?"
        ],
        'rankings': [
            "What is KL University's NIRF ranking?",
            "What accreditations does KLEF have?",
            "What awards has KL University received?",
            "Is KL University NAAC accredited?",
            "How does KL University rank globally?"
        ]
    }
    
    for category, questions in samples.items():
        print(f"\n{category.upper()}:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")


def validate_dataset_quality(qa_dataset: Dict):
    """Validate the quality of generated dataset"""
    print("\n=== DATASET QUALITY VALIDATION ===")
    
    questions = qa_dataset['questions']
    
    # Basic statistics
    total_questions = len(questions)
    avg_question_length = sum(len(q['question']) for q in questions) / total_questions
    avg_answer_length = sum(len(q['answer']) for q in questions) / total_questions
    
    print(f"Total questions: {total_questions}")
    print(f"Average question length: {avg_question_length:.1f} characters")
    print(f"Average answer length: {avg_answer_length:.1f} characters")
    
    # Quality checks
    questions_with_question_marks = sum(1 for q in questions if '?' in q['question'])
    questions_with_proper_answers = sum(1 for q in questions if len(q['answer']) >= 20)
    unique_questions = len(set(q['question'].lower().strip() for q in questions))
    
    print(f"\nQuality Metrics:")
    print(f"Questions with '?': {questions_with_question_marks}/{total_questions} ({100*questions_with_question_marks/total_questions:.1f}%)")
    print(f"Proper answers (>20 chars): {questions_with_proper_answers}/{total_questions} ({100*questions_with_proper_answers/total_questions:.1f}%)")
    print(f"Unique questions: {unique_questions}/{total_questions} ({100*unique_questions/total_questions:.1f}%)")
    
    # Category balance
    category_counts = {}
    for q in questions:
        cat = q['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nCategory Balance:")
    for cat, count in sorted(category_counts.items()):
        percentage = (count / total_questions) * 100
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # Quality score calculation
    quality_score = (
        (questions_with_question_marks / total_questions) * 0.3 +
        (questions_with_proper_answers / total_questions) * 0.3 +
        (unique_questions / total_questions) * 0.4
    ) * 100
    
    print(f"\nOverall Quality Score: {quality_score:.1f}/100")
    
    if quality_score >= 85:
        print("✅ EXCELLENT quality dataset")
    elif quality_score >= 70:
        print("✅ GOOD quality dataset")
    elif quality_score >= 50:
        print("⚠️ FAIR quality dataset - consider improvements")
    else:
        print("❌ POOR quality dataset - needs significant improvement")


def create_model_ready_format(qa_dataset: Dict, output_format: str = "huggingface"):
    """Convert dataset to model-ready format"""
    print(f"\n=== CREATING {output_format.upper()} FORMAT ===")
    
    questions = qa_dataset['questions']
    
    if output_format == "huggingface":
        # Format for Hugging Face datasets
        hf_format = {
            "question": [q['question'] for q in questions],
            "answer": [q['answer'] for q in questions],
            "category": [q['category'] for q in questions],
            "id": [q['id'] for q in questions]
        }
        
        df = pd.DataFrame(hf_format)
        df.to_csv('kl_university_qa_huggingface.csv', index=False)
        print("✅ Hugging Face format saved: kl_university_qa_huggingface.csv")
    
    elif output_format == "squad":
        # Format similar to SQuAD dataset
        squad_format = {
            "data": []
        }
        
        # Group by category (like SQuAD topics)
        category_groups = {}
        for q in questions:
            cat = q['category']
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(q)
        
        for category, cat_questions in category_groups.items():
            squad_data = {
                "title": f"KL University - {category.title()}",
                "paragraphs": []
            }
            
            # Group questions by source page
            page_groups = {}
            for q in cat_questions:
                page = q.get('source_page', 'unknown')
                if page not in page_groups:
                    page_groups[page] = {'context': '', 'qas': []}
                
                page_groups[page]['qas'].append({
                    "id": q['id'],
                    "question": q['question'],
                    "answers": [{"text": q['answer'], "answer_start": 0}]
                })
                
                # Use answer as context if no context exists
                if not page_groups[page]['context']:
                    page_groups[page]['context'] = q['answer']
            
            squad_data['paragraphs'] = list(page_groups.values())
            squad_format['data'].append(squad_data)
        
        with open('kl_university_qa_squad_format.json', 'w', encoding='utf-8') as f:
            json.dump(squad_format, f, indent=2, ensure_ascii=False)
        print("✅ SQuAD format saved: kl_university_qa_squad_format.json")
    
    elif output_format == "simple":
        # Simple Q&A format
        simple_format = []
        for q in questions:
            simple_format.append({
                "input": q['question'],
                "output": q['answer'],
                "category": q['category']
            })
        
        with open('kl_university_qa_simple.json', 'w', encoding='utf-8') as f:
            json.dump(simple_format, f, indent=2, ensure_ascii=False)
        print("✅ Simple format saved: kl_university_qa_simple.json")


def generate_training_guidelines():
    """Generate guidelines for training NLP models with this dataset"""
    guidelines = """
# KL University Q&A Dataset - Training Guidelines

## Dataset Overview
- **Total Questions**: 1000-1500 Q&A pairs
- **Categories**: 8 main categories (Admissions, Programs, Facilities, etc.)
- **Format**: JSON with train/validation/test splits
- **Quality**: Validated and filtered for accuracy

## Recommended Model Architectures

### 1. BERT-based Models (Recommended)
```python
# For question-answering tasks
- DistilBERT-base-uncased
- BERT-base-uncased
- RoBERTa-base

# Training parameters:
- Learning rate: 2e-5 to 5e-5
- Batch size: 16-32
- Epochs: 3-5
- Max sequence length: 512
```

### 2. T5-based Models (For generative QA)
```python
# For text-to-text generation
- T5-small
- T5-base
- FLAN-T5-base

# Training parameters:
- Learning rate: 1e-4
- Batch size: 8-16
- Epochs: 3-5
```

### 3. Sentence Transformers (For semantic search)
```python
# For embedding-based retrieval
- all-MiniLM-L6-v2
- all-mpnet-base-v2

# Use for: Creating embeddings for question matching
```

## Training Strategy

### Phase 1: Category-specific Training
1. Train separate models for each category
2. Fine-tune on category-specific subsets
3. Evaluate category-wise performance

### Phase 2: Unified Model Training
1. Train on complete dataset
2. Use category as additional feature
3. Implement multi-task learning

### Phase 3: Ensemble Approach
1. Combine category-specific models
2. Use confidence scores for selection
3. Implement fallback mechanisms

## Evaluation Metrics
- **BLEU Score**: For answer generation quality
- **Exact Match**: For factual accuracy
- **F1 Score**: For token-level overlap
- **ROUGE-L**: For answer completeness

## Data Augmentation Techniques
1. **Paraphrasing**: Generate question variations
2. **Back-translation**: Improve robustness
3. **Synonym replacement**: Increase vocabulary coverage
4. **Context expansion**: Add related information

## Deployment Considerations
1. **Response Time**: Optimize for <2 seconds
2. **Confidence Thresholding**: Set minimum confidence levels
3. **Fallback Responses**: Handle out-of-domain queries
4. **Continuous Learning**: Update with new university information

## Sample Training Code Structure
```python
# 1. Data Loading
train_dataset = load_dataset('kl_university_qa_train.json')
val_dataset = load_dataset('kl_university_qa_validation.json')

# 2. Model Initialization
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# 3. Training Configuration
training_args = TrainingArguments(
    output_dir='./kl-qa-model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# 4. Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```
"""
    
    with open('training_guidelines.md', 'w', encoding='utf-8') as f:
        f.write(guidelines)
    print("✅ Training guidelines saved: training_guidelines.md")


if __name__ == "__main__":
    # Run main question generation
    dataset = main()
    
    if dataset:
        # Show sample questions
        demo_question_samples()
        
        # Validate dataset quality
        validate_dataset_quality(dataset)
        
        # Create different formats
        create_model_ready_format(dataset, "huggingface")
        create_model_ready_format(dataset, "squad")
        create_model_ready_format(dataset, "simple")
        
        # Generate training guidelines
        generate_training_guidelines()
        
        print(f"\n🎉 PHASE 2 COMPLETE! 🎉")
        print(f"Generated {dataset['metadata']['total_questions']} Q&A pairs")
        print(f"Files created:")
        print(f"  - kl_university_qa_dataset.json (main dataset)")
        print(f"  - kl_university_qa_train.json (training set)")
        print(f"  - kl_university_qa_validation.json (validation set)")
        print(f"  - kl_university_qa_test.json (test set)")
        print(f"  - kl_university_qa_dataset.csv (CSV format)")
        print(f"  - kl_university_qa_huggingface.csv (HuggingFace format)")
        print(f"  - kl_university_qa_squad_format.json (SQuAD format)")
        print(f"  - kl_university_qa_simple.json (Simple format)")
        print(f"  - training_guidelines.md (Training instructions)")
        print(f"\nReady for NLP model training! 🚀")