import re
import json
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Optional imports with fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
except ImportError:
    print("NLTK not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk
    NLTK_AVAILABLE = True

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    print("spaCy not available. Some advanced features will be disabled.")
    SPACY_AVAILABLE = False

class UniversityDataPreprocessor:
    """
    Automated data preparation for KL University content
    """
    
    def __init__(self):
        # Load spaCy model for NLP processing (optional)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ spaCy model loaded successfully")
            except IOError:
                print("⚠ spaCy model 'en_core_web_sm' not found.")
                print("Install with: python -m spacy download en_core_web_sm")
                print("Continuing without spaCy (basic functionality available)")
        else:
            print("⚠ spaCy not available. Advanced NLP features disabled.")
        
        # Define content categories
        self.categories = {
            'admissions': ['admission', 'entrance', 'exam', 'eligibility', 'fee', 'application', 'kleee', 'klmat'],
            'programs': ['btech', 'mtech', 'program', 'course', 'degree', 'specialization', 'curriculum'],
            'facilities': ['hostel', 'library', 'lab', 'infrastructure', 'campus', 'facility'],
            'rankings': ['rank', 'nirf', 'award', 'recognition', 'accreditation', 'naac'],
            'faculty': ['faculty', 'professor', 'research', 'publication', 'phd'],
            'student_life': ['sport', 'club', 'placement', 'activity', 'student'],
            'administration': ['management', 'chancellor', 'vice-chancellor', 'governance']
        }
        
        # Patterns for cleaning
        self.cleanup_patterns = [
            (r'={2,}.*?={2,}', ''),  # Remove === headers ===
            (r'URL:.*?(?=\n)', ''),  # Remove URLs
            (r'Word Count:.*?(?=\n)', ''),  # Remove word counts
            (r'Scraped At:.*?(?=\n)', ''),  # Remove scrape timestamps
            (r'Content Types:.*?(?=\n)', ''),  # Remove content types
            (r'Â©.*?All Rights Reserved\.?', ''),  # Remove copyright
            (r'\[PHONE\]', '[PHONE_NUMBER]'),  # Anonymize phone numbers
            (r'\[EMAIL\]', '[EMAIL_ADDRESS]'),  # Anonymize emails
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'\n\s*\n', '\n'),  # Multiple newlines to single
        ]

    def load_data(self, file_path: str) -> str:
        """Load the text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()

    def clean_text(self, text: str) -> str:
        """Clean and standardize the text"""
        cleaned_text = text
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def extract_pages(self, text: str) -> List[Dict]:
        """Extract individual pages from the document"""
        pages = []
        
        # Split by page markers
        page_pattern = r'=== Page (\d+):(.*?) ==='
        page_splits = re.split(page_pattern, text)
        
        for i in range(1, len(page_splits), 3):
            if i + 2 < len(page_splits):
                page_num = page_splits[i]
                page_title = page_splits[i + 1].strip()
                page_content = page_splits[i + 2].strip()
                
                pages.append({
                    'page_number': int(page_num),
                    'title': page_title,
                    'content': self.clean_text(page_content),
                    'url': self.extract_url(page_content)
                })
        
        return pages

    def extract_url(self, content: str) -> str:
        """Extract URL from page content"""
        url_pattern = r'URL: (https?://[^\s]+)'
        match = re.search(url_pattern, content)
        return match.group(1) if match else ""

    def categorize_content(self, content: str) -> List[str]:
        """Categorize content based on keywords"""
        content_lower = content.lower()
        categories_found = []
        
        for category, keywords in self.categories.items():
            if any(keyword in content_lower for keyword in keywords):
                categories_found.append(category)
        
        return categories_found if categories_found else ['general']

    def extract_key_information(self, content: str) -> Dict:
        """Extract structured information from content"""
        info = {
            'fees': [],
            'dates': [],
            'numbers': [],
            'programs': [],
            'contact_info': [],
            'requirements': []
        }
        
        # Extract fees (Rs. patterns)
        fee_patterns = [
            r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'INR\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+)\s*(?:lakhs?|crores?)'
        ]
        
        for pattern in fee_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            info['fees'].extend(matches)
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            info['dates'].extend(matches)
        
        # Extract programs (B.Tech, M.Tech, etc.)
        program_pattern = r'\b(?:B\.?Tech|M\.?Tech|B\.?Sc|M\.?Sc|MBA|BBA|Ph\.?D|B\.?Arch|B\.?Pharmacy)\b[^.\n]*'
        info['programs'] = re.findall(program_pattern, content, re.IGNORECASE)
        
        # Extract percentages and numbers
        number_patterns = [
            r'(\d+)%',
            r'(\d+)\+?\s*(?:students?|faculty|courses?|labs?)',
            r'rank(?:ed)?\s*(\d+)',
            r'(\d+)(?:st|nd|rd|th)\s*rank'
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            info['numbers'].extend(matches)
        
        return info

    def segment_content(self, content: str) -> List[Dict]:
        """Segment content into logical sections"""
        segments = []
        
        # Split by common section headers
        section_patterns = [
            r'(?:^|\n)([A-Z][A-Z\s&-]+:?)(?=\n)',  # ALL CAPS headers
            r'(?:^|\n)(Vision|Mission|Objectives?|Features?|About|Overview)(?=\n|\s)',
            r'(?:^|\n)(\d+\.\s+[A-Z][^.\n]*)',  # Numbered sections
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                header = match.group(1).strip()
                start_pos = match.end()
                
                # Find content until next header or end
                next_match = re.search(pattern, content[start_pos:], re.MULTILINE | re.IGNORECASE)
                end_pos = start_pos + next_match.start() if next_match else len(content)
                
                section_content = content[start_pos:end_pos].strip()
                
                if len(section_content) > 50:  # Only include substantial sections
                    segments.append({
                        'header': header,
                        'content': section_content,
                        'word_count': len(section_content.split())
                    })
        
        # If no clear sections found, split by paragraphs
        if not segments:
            paragraphs = content.split('\n\n')
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 100:
                    segments.append({
                        'header': f'Section {i+1}',
                        'content': para.strip(),
                        'word_count': len(para.split())
                    })
        
        return segments

    def process_document(self, file_path: str) -> Dict:
        """Main processing function"""
        print("Loading document...")
        raw_text = self.load_data(file_path)
        
        print("Extracting pages...")
        pages = self.extract_pages(raw_text)
        
        print("Processing pages...")
        processed_pages = []
        
        for page in pages:
            # Categorize content
            categories = self.categorize_content(page['content'])
            
            # Extract key information
            key_info = self.extract_key_information(page['content'])
            
            # Segment content
            segments = self.segment_content(page['content'])
            
            processed_page = {
                'page_number': page['page_number'],
                'title': page['title'],
                'url': page['url'],
                'content': page['content'],
                'categories': categories,
                'key_information': key_info,
                'segments': segments,
                'word_count': len(page['content'].split())
            }
            
            processed_pages.append(processed_page)
        
        # Create summary statistics
        summary = self.create_summary(processed_pages)
        
        return {
            'pages': processed_pages,
            'summary': summary,
            'total_pages': len(processed_pages)
        }

    def create_summary(self, pages: List[Dict]) -> Dict:
        """Create summary statistics"""
        category_counts = defaultdict(int)
        total_words = 0
        program_mentions = set()
        
        for page in pages:
            total_words += page['word_count']
            for category in page['categories']:
                category_counts[category] += 1
            program_mentions.update(page['key_information']['programs'])
        
        return {
            'total_words': total_words,
            'category_distribution': dict(category_counts),
            'unique_programs': list(program_mentions)[:20],  # Top 20
            'avg_words_per_page': total_words / len(pages) if pages else 0
        }

    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"Processed data saved to: {output_path}")

    def export_to_csv(self, processed_data: Dict, output_path: str):
        """Export processed data to CSV for easy analysis"""
        rows = []
        
        for page in processed_data['pages']:
            for segment in page['segments']:
                rows.append({
                    'page_number': page['page_number'],
                    'page_title': page['title'],
                    'categories': '|'.join(page['categories']),
                    'segment_header': segment['header'],
                    'segment_content': segment['content'][:500] + '...' if len(segment['content']) > 500 else segment['content'],
                    'word_count': segment['word_count'],
                    'url': page['url']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"CSV export saved to: {output_path}")


# Example usage and demonstration
def main():
    """Main execution function"""
    # Initialize preprocessor
    preprocessor = UniversityDataPreprocessor()
    
    # Process the document
    try:
        # Replace 'all_content.txt' with your actual file path
        processed_data = preprocessor.process_document('all_content.txt')
        
        # Save processed data
        preprocessor.save_processed_data(processed_data, 'processed_university_data.json')
        
        # Export to CSV for analysis
        preprocessor.export_to_csv(processed_data, 'university_data_segments.csv')
        
        # Print summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total pages processed: {processed_data['total_pages']}")
        print(f"Total words: {processed_data['summary']['total_words']:,}")
        print(f"Average words per page: {processed_data['summary']['avg_words_per_page']:.0f}")
        
        print("\nCategory Distribution:")
        for category, count in processed_data['summary']['category_distribution'].items():
            print(f"  {category}: {count} pages")
        
        print(f"\nSample programs found: {processed_data['summary']['unique_programs'][:5]}")
        
    except FileNotFoundError:
        print("Error: 'all_content.txt' file not found. Please check the file path.")
    except Exception as e:
        print(f"Error processing document: {str(e)}")


if __name__ == "__main__":
    main()


# Additional utility functions for specific preprocessing tasks

def extract_admission_data(processed_data: Dict) -> List[Dict]:
    """Extract admission-specific information"""
    admission_data = []
    
    for page in processed_data['pages']:
        if 'admissions' in page['categories']:
            for segment in page['segments']:
                if any(keyword in segment['content'].lower() for keyword in ['fee', 'eligibility', 'exam', 'admission']):
                    admission_data.append({
                        'source_page': page['page_number'],
                        'title': page['title'],
                        'content': segment['content'],
                        'fees': page['key_information']['fees'],
                        'dates': page['key_information']['dates']
                    })
    
    return admission_data

def extract_program_data(processed_data: Dict) -> List[Dict]:
    """Extract program-specific information"""
    program_data = []
    
    for page in processed_data['pages']:
        if 'programs' in page['categories']:
            program_data.append({
                'source_page': page['page_number'],
                'title': page['title'],
                'programs': page['key_information']['programs'],
                'content': page['content']
            })
    
    return program_data

def create_category_specific_datasets(processed_data: Dict) -> Dict:
    """Create separate datasets for each category"""
    category_datasets = {}
    
    for category in ['admissions', 'programs', 'facilities', 'rankings', 'faculty', 'student_life']:
        category_pages = [page for page in processed_data['pages'] if category in page['categories']]
        category_datasets[category] = {
            'pages': category_pages,
            'page_count': len(category_pages),
            'total_words': sum(page['word_count'] for page in category_pages)
        }
    
    return category_datasets