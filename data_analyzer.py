import json
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud not available - word cloud visualization will be skipped")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Try to download required data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('vader_lexicon')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available - some advanced features will be limited")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - topic modeling and clustering will be limited")

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Textstat not available - readability analysis will be limited")

class ScrapedDataAnalyzer:
    def __init__(self, json_file_path: str):
        """Initialize analyzer with scraped data"""
        self.json_file_path = json_file_path
        self.data = self.load_data()
        self.df = self.create_dataframe()
        self.analysis_results = {}
        
        # Initialize components based on available libraries
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
                self.sentiment_analyzer = None
        else:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.sentiment_analyzer = None
        
        print(f"Loaded {len(self.df)} pages for analysis")
    
    def load_data(self) -> Dict:
        """Load scraped data from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {self.json_file_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file {self.json_file_path}")
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convert scraped data to pandas DataFrame"""
        pages = self.data.get('scraped_pages', [])
        df = pd.DataFrame(pages)
        
        if not df.empty:
            # Add derived columns
            df['content_length'] = df['content'].str.len()
            df['sentence_count'] = df['content'].apply(self.count_sentences)
            df['avg_sentence_length'] = df['word_count'] / df['sentence_count'].replace(0, 1)
            
            # Handle datetime conversion safely
            try:
                df['scraped_at'] = pd.to_datetime(df['scraped_at'])
            except:
                df['scraped_at'] = pd.to_datetime('now')
            
            df['domain'] = df['url'].apply(lambda x: x.split('/')[2] if len(x.split('/')) > 2 else '')
            df['url_depth'] = df['url'].apply(lambda x: len(x.split('/')) - 3)
            
        return df
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if not text:
            return 0
        
        if NLTK_AVAILABLE:
            try:
                return len(sent_tokenize(text))
            except:
                pass
        
        # Fallback method
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        
        # Fallback method
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def analyze_content_quality(self) -> Dict:
        """Analyze content quality metrics"""
        print("Analyzing content quality...")
        
        quality_metrics = {
            'readability': self.analyze_readability(),
            'language_diversity': self.analyze_language_diversity(),
            'content_structure': self.analyze_content_structure(),
            'information_density': self.analyze_information_density(),
            'completeness': self.analyze_completeness()
        }
        
        return quality_metrics
    
    def analyze_readability(self) -> Dict:
        """Analyze text readability using various metrics"""
        if not TEXTSTAT_AVAILABLE:
            return {"error": "Textstat library not available"}
        
        readability_scores = {
            'flesch_reading_ease': [],
            'flesch_kincaid_grade': [],
            'gunning_fog': [],
            'automated_readability_index': []
        }
        
        sample_size = min(50, len(self.df))
        for content in self.df['content'].head(sample_size):
            if len(str(content)) > 100:
                try:
                    readability_scores['flesch_reading_ease'].append(textstat.flesch_reading_ease(content))
                    readability_scores['flesch_kincaid_grade'].append(textstat.flesch_kincaid_grade(content))
                    readability_scores['gunning_fog'].append(textstat.gunning_fog(content))
                    readability_scores['automated_readability_index'].append(textstat.automated_readability_index(content))
                except:
                    continue
        
        # Calculate averages
        avg_scores = {}
        for metric, scores in readability_scores.items():
            if scores:
                avg_scores[metric] = {
                    'average': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        return avg_scores
    
    def analyze_language_diversity(self) -> Dict:
        """Analyze vocabulary diversity and language patterns"""
        all_text = ' '.join(self.df['content'].astype(str))
        words = self.tokenize_text(all_text)
        words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 2]
        
        # Vocabulary metrics
        vocab_size = len(set(words))
        total_words = len(words)
        type_token_ratio = vocab_size / total_words if total_words > 0 else 0
        
        # Most common words
        word_freq = Counter(words)
        most_common = word_freq.most_common(50)
        
        # Language patterns
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        return {
            'vocabulary_size': vocab_size,
            'total_words': total_words,
            'type_token_ratio': type_token_ratio,
            'average_word_length': avg_word_length,
            'most_common_words': most_common[:20],
            'vocabulary_richness': self.calculate_vocabulary_richness(words)
        }
    
    def calculate_vocabulary_richness(self, words: List[str]) -> Dict:
        """Calculate vocabulary richness metrics"""
        if not words:
            return {}
        
        word_freq = Counter(words)
        freq_dist = list(word_freq.values())
        
        # Hapax legomena (words appearing once)
        hapax = sum(1 for freq in freq_dist if freq == 1)
        hapax_ratio = hapax / len(word_freq) if word_freq else 0
        
        # Dis legomena (words appearing twice)
        dis = sum(1 for freq in freq_dist if freq == 2)
        dis_ratio = dis / len(word_freq) if word_freq else 0
        
        return {
            'hapax_legomena_count': hapax,
            'hapax_ratio': hapax_ratio,
            'dis_legomena_count': dis,
            'dis_ratio': dis_ratio,
            'frequency_distribution': dict(Counter(freq_dist).most_common(10))
        }
    
    def analyze_content_structure(self) -> Dict:
        """Analyze content structure and organization"""
        structure_metrics = {
            'avg_sentences_per_page': 0,
            'avg_words_per_sentence': 0,
            'content_type_distribution': {},
            'title_quality': self.analyze_title_quality()
        }
        
        # Calculate averages
        if not self.df.empty:
            structure_metrics['avg_sentences_per_page'] = self.df['sentence_count'].mean()
            structure_metrics['avg_words_per_sentence'] = self.df['avg_sentence_length'].mean()
            
            # Content type distribution
            all_content_types = []
            for types in self.df['content_type']:
                if isinstance(types, list):
                    all_content_types.extend(types)
                elif isinstance(types, str) and types:
                    all_content_types.extend([t.strip() for t in types.split(';')])
            
            structure_metrics['content_type_distribution'] = dict(Counter(all_content_types))
        
        return structure_metrics
    
    def analyze_title_quality(self) -> Dict:
        """Analyze quality of page titles"""
        titles = self.df['title'].dropna()
        
        if titles.empty:
            return {}
        
        title_lengths = titles.str.len()
        word_counts = titles.str.split().str.len()
        
        return {
            'avg_title_length': title_lengths.mean(),
            'avg_title_words': word_counts.mean(),
            'title_length_distribution': {
                'min': title_lengths.min(),
                'max': title_lengths.max(),
                'median': title_lengths.median()
            },
            'empty_titles': self.df['title'].isna().sum(),
            'duplicate_titles': len(titles) - len(titles.unique())
        }
    
    def analyze_information_density(self) -> Dict:
        """Analyze information density and content value"""
        density_metrics = {
            'words_per_page_stats': self.df['word_count'].describe().to_dict(),
            'content_length_stats': self.df['content_length'].describe().to_dict(),
            'pages_by_content_size': self.categorize_pages_by_size(),
            'information_density_score': self.calculate_info_density_score()
        }
        
        return density_metrics
    
    def categorize_pages_by_size(self) -> Dict:
        """Categorize pages by content size"""
        word_counts = self.df['word_count']
        
        categories = {
            'very_short': (word_counts < 100).sum(),
            'short': ((word_counts >= 100) & (word_counts < 300)).sum(),
            'medium': ((word_counts >= 300) & (word_counts < 800)).sum(),
            'long': ((word_counts >= 800) & (word_counts < 2000)).sum(),
            'very_long': (word_counts >= 2000).sum()
        }
        
        total = sum(categories.values())
        percentages = {k: (v/total*100) if total > 0 else 0 for k, v in categories.items()}
        
        return {
            'counts': categories,
            'percentages': percentages
        }
    
    def calculate_info_density_score(self) -> float:
        """Calculate overall information density score (0-100)"""
        if self.df.empty:
            return 0
        
        # Factors for information density
        avg_words = self.df['word_count'].mean()
        avg_sentences = self.df['sentence_count'].mean()
        avg_sentence_length = self.df['avg_sentence_length'].mean()
        
        # Normalize scores (higher is better)
        word_score = min(avg_words / 500, 1) * 30  # Max 30 points
        sentence_score = min(avg_sentences / 20, 1) * 25  # Max 25 points
        length_score = min(avg_sentence_length / 15, 1) * 20  # Max 20 points
        
        # Content diversity score
        unique_content = len(self.df['page_hash'].unique()) / len(self.df) * 25  # Max 25 points
        
        return word_score + sentence_score + length_score + unique_content
    
    def analyze_completeness(self) -> Dict:
        """Analyze data completeness and quality issues"""
        completeness = {
            'total_pages': len(self.df),
            'missing_data': {
                'empty_content': (self.df['content'].str.len() < 10).sum(),
                'missing_titles': self.df['title'].isna().sum(),
                'missing_metadata': self.df['metadata'].isna().sum() if 'metadata' in self.df.columns else 0,
            },
            'duplicate_content': len(self.df) - len(self.df['page_hash'].unique()) if 'page_hash' in self.df.columns else 0,
            'failed_scrapes': len(self.data.get('failed_urls', [])),
            'success_rate': len(self.df) / (len(self.df) + len(self.data.get('failed_urls', []))) * 100 if len(self.df) > 0 else 0
        }
        
        return completeness
    
    def analyze_topic_distribution(self, n_topics: int = 8) -> Dict:
        """Analyze topic distribution using LDA if available"""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available for topic analysis"}
        
        print("Analyzing topic distribution...")
        
        # Prepare text data
        documents = self.df['content'].dropna().astype(str).tolist()
        if len(documents) < 2:
            return {"error": "Not enough documents for topic analysis"}
        
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                max_df=0.95,
                min_df=2,
                ngram_range=(1, 2)
            )
            
            doc_term_matrix = vectorizer.fit_transform(documents)
            
            # Perform LDA
            n_topics = min(n_topics, len(documents), 10)
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weight': float(topic.max())
                })
            
            return {
                'topics': topics,
                'coherence_score': float(np.mean(np.max(lda.transform(doc_term_matrix), axis=1))),
                'topic_distribution': lda.transform(doc_term_matrix).mean(axis=0).tolist()
            }
        
        except Exception as e:
            return {"error": f"Topic analysis failed: {str(e)}"}
    
    def analyze_content_similarity(self) -> Dict:
        """Analyze content similarity and clustering"""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available for similarity analysis"}
        
        print("Analyzing content similarity...")
        
        documents = self.df['content'].dropna().astype(str).tolist()[:50]  # Sample for performance
        if len(documents) < 2:
            return {"error": "Not enough documents for similarity analysis"}
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=300,
                stop_words='english',
                max_df=0.95,
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate average similarity
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            avg_similarity = similarity_matrix[mask].mean()
            
            # Perform clustering
            n_clusters = min(5, len(documents) // 3, 8)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(tfidf_matrix)
                cluster_distribution = dict(Counter(clusters))
            else:
                cluster_distribution = {0: len(documents)}
            
            return {
                'average_similarity': float(avg_similarity),
                'similarity_distribution': {
                    'low_similarity': float(np.sum(similarity_matrix < 0.2) / similarity_matrix.size),
                    'medium_similarity': float(np.sum((similarity_matrix >= 0.2) & (similarity_matrix < 0.7)) / similarity_matrix.size),
                    'high_similarity': float(np.sum(similarity_matrix >= 0.7) / similarity_matrix.size)
                },
                'clustering': {
                    'n_clusters': n_clusters,
                    'cluster_distribution': cluster_distribution
                }
            }
        
        except Exception as e:
            return {"error": f"Similarity analysis failed: {str(e)}"}
    
    def analyze_sentiment_distribution(self) -> Dict:
        """Analyze sentiment distribution in content"""
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analyzer not available"}
        
        print("Analyzing sentiment distribution...")
        
        sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
        sentiment_scores = []
        
        sample_size = min(50, len(self.df))
        for content in self.df['content'].head(sample_size):
            try:
                content_str = str(content)[:1000]  # First 1000 chars
                scores = self.sentiment_analyzer.polarity_scores(content_str)
                sentiment_scores.append(scores)
                
                if scores['compound'] >= 0.05:
                    sentiments['positive'] += 1
                elif scores['compound'] <= -0.05:
                    sentiments['negative'] += 1
                else:
                    sentiments['neutral'] += 1
            except:
                continue
        
        total = sum(sentiments.values())
        sentiment_percentages = {k: (v/total*100) if total > 0 else 0 for k, v in sentiments.items()}
        
        if sentiment_scores:
            avg_scores = {
                'compound': np.mean([s['compound'] for s in sentiment_scores]),
                'positive': np.mean([s['pos'] for s in sentiment_scores]),
                'negative': np.mean([s['neg'] for s in sentiment_scores]),
                'neutral': np.mean([s['neu'] for s in sentiment_scores])
            }
        else:
            avg_scores = {}
        
        return {
            'sentiment_distribution': sentiments,
            'sentiment_percentages': sentiment_percentages,
            'average_scores': avg_scores
        }
    
    def evaluate_training_suitability(self) -> Dict:
        """Evaluate how suitable the data is for different AI applications"""
        print("Evaluating training suitability...")
        
        # Get analysis results
        quality = self.analyze_content_quality()
        topics = self.analyze_topic_distribution()
        similarity = self.analyze_content_similarity()
        sentiment = self.analyze_sentiment_distribution()
        
        # Calculate suitability scores for different applications
        suitability = {
            'chatbot_training': self.evaluate_chatbot_suitability(quality, topics, similarity),
            'qa_system': self.evaluate_qa_suitability(quality, topics),
            'language_model': self.evaluate_lm_suitability(quality, similarity),
            'content_classification': self.evaluate_classification_suitability(topics, similarity),
            'sentiment_analysis': self.evaluate_sentiment_suitability(sentiment),
            'summarization': self.evaluate_summarization_suitability(quality, similarity)
        }
        
        return suitability
    
    def evaluate_chatbot_suitability(self, quality: Dict, topics: Dict, similarity: Dict) -> Dict:
        """Evaluate suitability for chatbot training"""
        score = 0
        factors = []
        
        # Content diversity (30 points)
        if not topics.get('error') and len(topics.get('topics', [])) > 5:
            score += 30
            factors.append("Good topic diversity")
        elif not topics.get('error') and len(topics.get('topics', [])) > 2:
            score += 20
            factors.append("Moderate topic diversity")
        else:
            factors.append("Limited topic diversity")
        
        # Content size (25 points)
        info_density = quality.get('information_density', {}).get('information_density_score', 0)
        if info_density > 70:
            score += 25
            factors.append("High information density")
        elif info_density > 40:
            score += 15
            factors.append("Moderate information density")
        else:
            factors.append("Low information density")
        
        # Language quality (25 points)
        readability = quality.get('readability', {})
        if readability and not readability.get('error'):
            avg_ease = readability.get('flesch_reading_ease', {}).get('average', 0)
            if avg_ease > 60:
                score += 25
                factors.append("Good readability")
            elif avg_ease > 30:
                score += 15
                factors.append("Moderate readability")
        else:
            score += 10
            factors.append("Readability analysis limited")
        
        # Content uniqueness (20 points)
        if not similarity.get('error'):
            avg_sim = similarity.get('average_similarity', 1)
            if avg_sim < 0.3:
                score += 20
                factors.append("High content uniqueness")
            elif avg_sim < 0.6:
                score += 10
                factors.append("Moderate content uniqueness")
            else:
                factors.append("High content similarity")
        else:
            score += 10
            factors.append("Similarity analysis limited")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_chatbot_recommendations(score)
        }
    
    def evaluate_qa_suitability(self, quality: Dict, topics: Dict) -> Dict:
        """Evaluate suitability for Q&A system training"""
        score = 0
        factors = []
        
        # Content structure (40 points)
        structure = quality.get('content_structure', {})
        avg_sentences = structure.get('avg_sentences_per_page', 0)
        if avg_sentences > 10:
            score += 40
            factors.append("Well-structured content with multiple sentences")
        elif avg_sentences > 5:
            score += 25
            factors.append("Moderately structured content")
        else:
            factors.append("Limited content structure")
        
        # Information density (30 points)
        info_density = quality.get('information_density', {}).get('information_density_score', 0)
        if info_density > 60:
            score += 30
            factors.append("High information density")
        elif info_density > 30:
            score += 20
            factors.append("Moderate information density")
        
        # Topic coverage (30 points)
        if not topics.get('error') and len(topics.get('topics', [])) > 6:
            score += 30
            factors.append("Excellent topic coverage")
        elif not topics.get('error') and len(topics.get('topics', [])) > 3:
            score += 20
            factors.append("Good topic coverage")
        else:
            factors.append("Limited topic coverage")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_qa_recommendations(score)
        }
    
    def evaluate_lm_suitability(self, quality: Dict, similarity: Dict) -> Dict:
        """Evaluate suitability for language model training"""
        score = 0
        factors = []
        
        # Vocabulary richness (35 points)
        lang_div = quality.get('language_diversity', {})
        ttr = lang_div.get('type_token_ratio', 0)
        if ttr > 0.15:
            score += 35
            factors.append("Rich vocabulary diversity")
        elif ttr > 0.08:
            score += 25
            factors.append("Good vocabulary diversity")
        else:
            factors.append("Limited vocabulary diversity")
        
        # Content volume (35 points)
        total_words = lang_div.get('total_words', 0)
        if total_words > 100000:
            score += 35
            factors.append("Large text corpus")
        elif total_words > 50000:
            score += 25
            factors.append("Medium text corpus")
        elif total_words > 10000:
            score += 15
            factors.append("Small text corpus")
        else:
            factors.append("Very small text corpus")
        
        # Language quality (30 points)
        readability = quality.get('readability', {})
        if readability and not readability.get('error'):
            avg_grade = readability.get('flesch_kincaid_grade', {}).get('average', 20)
            if 8 <= avg_grade <= 14:
                score += 30
                factors.append("Appropriate reading level")
            elif avg_grade <= 16:
                score += 20
                factors.append("Acceptable reading level")
            else:
                score += 10
                factors.append("Complex reading level")
        else:
            score += 15
            factors.append("Reading level analysis limited")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_lm_recommendations(score)
        }
    
    def evaluate_classification_suitability(self, topics: Dict, similarity: Dict) -> Dict:
        """Evaluate suitability for content classification"""
        score = 0
        factors = []
        
        # Topic separation (50 points)
        if not topics.get('error'):
            n_topics = len(topics.get('topics', []))
            if n_topics > 6:
                score += 50
                factors.append("Excellent topic separation")
            elif n_topics > 3:
                score += 35
                factors.append("Good topic separation")
            elif n_topics > 1:
                score += 20
                factors.append("Basic topic separation")
            else:
                factors.append("Limited topic separation")
        else:
            score += 10
            factors.append("Topic analysis limited")
        
        # Content distinctiveness (50 points)
        if not similarity.get('error'):
            avg_sim = similarity.get('average_similarity', 1)
            if avg_sim < 0.4:
                score += 50
                factors.append("Highly distinct content")
            elif avg_sim < 0.7:
                score += 35
                factors.append("Moderately distinct content")
            else:
                score += 15
                factors.append("Similar content")
        else:
            score += 20
            factors.append("Similarity analysis limited")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_classification_recommendations(score)
        }
    
    def evaluate_sentiment_suitability(self, sentiment: Dict) -> Dict:
        """Evaluate suitability for sentiment analysis"""
        score = 0
        factors = []
        
        if sentiment.get('error'):
            return {
                'score': 30,
                'grade': 'D',
                'factors': ["Sentiment analysis not available"],
                'recommendations': ["Install NLTK for sentiment analysis capabilities"]
            }
        
        # Sentiment diversity (60 points)
        sent_dist = sentiment.get('sentiment_percentages', {})
        pos_pct = sent_dist.get('positive', 0)
        neg_pct = sent_dist.get('negative', 0)
        neu_pct = sent_dist.get('neutral', 0)
        
        # Check for balanced distribution
        if all(pct > 15 for pct in [pos_pct, neg_pct, neu_pct]):
            score += 60
            factors.append("Well-balanced sentiment distribution")
        elif pos_pct > 10 and neg_pct > 10:
            score += 40
            factors.append("Adequate sentiment diversity")
        else:
            score += 20
            factors.append("Limited sentiment diversity")
        
        # Content volume (40 points)
        total_analyzed = sum(sentiment.get('sentiment_distribution', {}).values())
        if total_analyzed > 30:
            score += 40
            factors.append("Sufficient content for sentiment analysis")
        elif total_analyzed > 15:
            score += 25
            factors.append("Moderate content volume")
        else:
            factors.append("Limited content volume")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_sentiment_recommendations(score)
        }
    
    def evaluate_summarization_suitability(self, quality: Dict, similarity: Dict) -> Dict:
        """Evaluate suitability for summarization training"""
        score = 0
        factors = []
        
        # Content length (40 points)
        info_density = quality.get('information_density', {})
        page_stats = info_density.get('words_per_page_stats', {})
        avg_words = page_stats.get('mean', 0)
        
        if avg_words > 500:
            score += 40
            factors.append("Good content length for summarization")
        elif avg_words > 200:
            score += 25
            factors.append("Adequate content length")
        else:
            factors.append("Short content may not need summarization")
        
        # Content structure (35 points)
        structure = quality.get('content_structure', {})
        avg_sentences = structure.get('avg_sentences_per_page', 0)
        if avg_sentences > 15:
            score += 35
            factors.append("Well-structured content with multiple sentences")
        elif avg_sentences > 8:
            score += 25
            factors.append("Moderately structured content")
        
        # Information density (25 points)
        density_score = info_density.get('information_density_score', 0)
        if density_score > 60:
            score += 25
            factors.append("High information density")
        elif density_score > 40:
            score += 15
            factors.append("Moderate information density")
        
        return {
            'score': score,
            'grade': self.score_to_grade(score),
            'factors': factors,
            'recommendations': self.get_summarization_recommendations(score)
        }
    
    def score_to_grade(self, score: int) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 45:
            return 'D+'
        elif score >= 40:
            return 'D'
        else:
            return 'F'
    
    def get_chatbot_recommendations(self, score: int) -> List[str]:
        """Get recommendations for chatbot training"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Consider scraping more diverse content sources",
                "Add FAQ pages and conversational content",
                "Include customer service interactions if available",
                "Supplement with external conversational datasets"
            ])
        elif score < 70:
            recommendations.extend([
                "Add more conversational and Q&A style content",
                "Include examples of natural dialogue",
                "Consider data augmentation techniques"
            ])
        else:
            recommendations.extend([
                "Data is suitable for chatbot training",
                "Consider fine-tuning pre-trained models",
                "Implement response quality evaluation metrics"
            ])
        
        return recommendations
    
    def get_qa_recommendations(self, score: int) -> List[str]:
        """Get recommendations for Q&A system training"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Add more structured content with clear information",
                "Include FAQ sections and documentation",
                "Focus on factual, informative content",
                "Consider manual Q&A pair creation"
            ])
        elif score < 70:
            recommendations.extend([
                "Improve content structure and organization",
                "Add more detailed explanations and examples",
                "Include question-answer pairs where possible"
            ])
        else:
            recommendations.extend([
                "Excellent for Q&A system development",
                "Consider implementing passage retrieval",
                "Use extractive and abstractive QA techniques"
            ])
        
        return recommendations
    
    def get_lm_recommendations(self, score: int) -> List[str]:
        """Get recommendations for language model training"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Increase corpus size significantly",
                "Add more diverse vocabulary and topics",
                "Consider combining with external text corpora",
                "Focus on data quality over quantity initially"
            ])
        elif score < 70:
            recommendations.extend([
                "Expand vocabulary through diverse content",
                "Increase total word count",
                "Add technical and domain-specific terminology"
            ])
        else:
            recommendations.extend([
                "Suitable for domain-specific language model training",
                "Consider transfer learning from pre-trained models",
                "Implement curriculum learning strategies"
            ])
        
        return recommendations
    
    def get_classification_recommendations(self, score: int) -> List[str]:
        """Get recommendations for classification training"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Create more distinct content categories",
                "Balance class distribution",
                "Add clear categorical labels",
                "Consider semi-supervised learning approaches"
            ])
        elif score < 70:
            recommendations.extend([
                "Improve topic separation",
                "Add more examples per category",
                "Consider hierarchical classification"
            ])
        else:
            recommendations.extend([
                "Excellent for classification tasks",
                "Consider multi-label classification",
                "Implement active learning for efficiency"
            ])
        
        return recommendations
    
    def get_sentiment_recommendations(self, score: int) -> List[str]:
        """Get recommendations for sentiment analysis"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Add more emotionally diverse content",
                "Include reviews, opinions, and subjective content",
                "Balance positive, negative, and neutral examples",
                "Consider domain-specific sentiment lexicons"
            ])
        elif score < 70:
            recommendations.extend([
                "Increase emotional diversity in content",
                "Add more subjective and opinion-based text",
                "Balance sentiment distribution"
            ])
        else:
            recommendations.extend([
                "Great for sentiment analysis training",
                "Consider aspect-based sentiment analysis",
                "Implement emotion detection capabilities"
            ])
        
        return recommendations
    
    def get_summarization_recommendations(self, score: int) -> List[str]:
        """Get recommendations for summarization training"""
        recommendations = []
        
        if score < 50:
            recommendations.extend([
                "Focus on longer, more detailed content",
                "Add articles, reports, and documentation",
                "Ensure content has clear main points",
                "Consider extractive summarization first"
            ])
        elif score < 70:
            recommendations.extend([
                "Increase average content length",
                "Add more structured documents",
                "Include clear topic sentences and conclusions"
            ])
        else:
            recommendations.extend([
                "Excellent for summarization tasks",
                "Consider both extractive and abstractive approaches",
                "Implement multi-document summarization"
            ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive analysis report"""
        print("Generating comprehensive analysis report...")
        
        # Run all analyses
        quality = self.analyze_content_quality()
        topics = self.analyze_topic_distribution()
        similarity = self.analyze_content_similarity()
        sentiment = self.analyze_sentiment_distribution()
        suitability = self.evaluate_training_suitability()
        
        # Compile comprehensive report
        report = {
            'dataset_overview': {
                'total_pages': len(self.df),
                'total_words': int(self.df['word_count'].sum()) if not self.df.empty else 0,
                'average_words_per_page': float(self.df['word_count'].mean()) if not self.df.empty else 0,
                'date_range': {
                    'earliest': self.df['scraped_at'].min().isoformat() if not self.df.empty else None,
                    'latest': self.df['scraped_at'].max().isoformat() if not self.df.empty else None
                },
                'unique_domains': int(self.df['domain'].nunique()) if not self.df.empty else 0,
                'failed_scrapes': len(self.data.get('failed_urls', []))
            },
            'content_quality_analysis': quality,
            'topic_analysis': topics,
            'similarity_analysis': similarity,
            'sentiment_analysis': sentiment,
            'training_suitability': suitability,
            'overall_assessment': self.generate_overall_assessment(suitability),
            'recommendations': self.generate_overall_recommendations(suitability),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def generate_overall_assessment(self, suitability: Dict) -> Dict:
        """Generate overall assessment of dataset quality"""
        scores = {app: data.get('score', 0) for app, data in suitability.items() if 'score' in data}
        
        if not scores:
            return {"error": "No suitability scores available"}
        
        avg_score = np.mean(list(scores.values()))
        best_application = max(scores.items(), key=lambda x: x[1])
        worst_application = min(scores.items(), key=lambda x: x[1])
        
        # Determine overall data quality tier
        if avg_score >= 75:
            quality_tier = "High Quality"
            tier_description = "Excellent for most AI applications"
        elif avg_score >= 60:
            quality_tier = "Good Quality"
            tier_description = "Suitable for many AI applications with some optimization"
        elif avg_score >= 45:
            quality_tier = "Fair Quality"
            tier_description = "May work for specific applications but needs improvement"
        else:
            quality_tier = "Poor Quality"
            tier_description = "Significant improvements needed before use"
        
        return {
            'overall_score': round(avg_score, 1),
            'overall_grade': self.score_to_grade(int(avg_score)),
            'quality_tier': quality_tier,
            'tier_description': tier_description,
            'best_application': best_application[0],
            'best_application_score': best_application[1],
            'worst_application': worst_application[0],
            'worst_application_score': worst_application[1],
            'score_distribution': scores
        }
    
    def generate_overall_recommendations(self, suitability: Dict) -> List[str]:
        """Generate overall recommendations for dataset improvement"""
        recommendations = []
        
        # Analyze common issues across applications
        low_scoring_apps = [app for app, data in suitability.items() 
                           if data.get('score', 0) < 60]
        
        if len(low_scoring_apps) > 3:
            recommendations.extend([
                "Consider expanding the dataset with more diverse content",
                "Focus on improving content quality and structure",
                "Add more domain-specific terminology and examples"
            ])
        
        # Check for specific common issues
        scores = {app: data.get('score', 0) for app, data in suitability.items()}
        
        if scores.get('language_model', 0) < 50:
            recommendations.append("Increase vocabulary diversity and total word count")
        
        if scores.get('chatbot_training', 0) < 50:
            recommendations.append("Add more conversational and interactive content")
        
        if scores.get('content_classification', 0) < 50:
            recommendations.append("Improve topic separation and content categorization")
        
        # General recommendations based on overall quality
        avg_score = np.mean(list(scores.values())) if scores else 0
        
        if avg_score < 60:
            recommendations.extend([
                "Consider data augmentation techniques",
                "Implement data cleaning and preprocessing",
                "Supplement with external high-quality datasets"
            ])
        else:
            recommendations.extend([
                "Dataset shows good potential for AI applications",
                "Consider specific fine-tuning for target applications",
                "Implement quality assurance measures"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def save_report(self, output_file: str = None) -> str:
        """Save comprehensive report to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'data_analysis_report_{timestamp}.json'
        
        report = self.generate_comprehensive_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to {output_file}")
        return output_file
    
    def create_visualizations(self, output_dir: str = "analysis_plots") -> Dict[str, str]:
        """Create visualization plots for the analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plots = {}
        
        try:
            # Set up matplotlib
            plt.style.use('default')
            
            # 1. Word count distribution
            plt.figure(figsize=(10, 6))
            self.df['word_count'].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of Word Counts per Page', fontsize=14, fontweight='bold')
            plt.xlabel('Word Count')
            plt.ylabel('Number of Pages')
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(output_dir, 'word_count_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['word_count_distribution'] = plot_path
            
            # 2. Content type distribution
            if not self.df.empty and 'content_type' in self.df.columns:
                all_types = []
                for types in self.df['content_type']:
                    if isinstance(types, list):
                        all_types.extend(types)
                    elif isinstance(types, str) and types:
                        all_types.extend([t.strip() for t in types.split(';')])
                
                if all_types:
                    type_counts = Counter(all_types)
                    plt.figure(figsize=(12, 6))
                    types, counts = zip(*type_counts.most_common(10))
                    plt.bar(types, counts, color='lightcoral', alpha=0.7)
                    plt.title('Content Type Distribution', fontsize=14, fontweight='bold')
                    plt.xlabel('Content Type')
                    plt.ylabel('Number of Pages')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    plot_path = os.path.join(output_dir, 'content_type_distribution.png')
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots['content_type_distribution'] = plot_path
            
            # 3. Page size categories
            quality = self.analyze_content_quality()
            size_categories = quality.get('information_density', {}).get('pages_by_content_size', {})
            if size_categories.get('percentages'):
                plt.figure(figsize=(10, 8))
                labels = list(size_categories['percentages'].keys())
                sizes = list(size_categories['percentages'].values())
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                plt.title('Page Size Distribution', fontsize=14, fontweight='bold')
                plot_path = os.path.join(output_dir, 'page_size_distribution.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['page_size_distribution'] = plot_path
            
            # 4. Training suitability scores
            suitability = self.evaluate_training_suitability()
            app_scores = {app.replace('_', ' ').title(): data.get('score', 0) 
                         for app, data in suitability.items() if 'score' in data}
            
            if app_scores:
                plt.figure(figsize=(12, 8))
                apps = list(app_scores.keys())
                scores = list(app_scores.values())
                colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]
                
                bars = plt.bar(apps, scores, color=colors, alpha=0.7)
                plt.title('Training Suitability Scores by Application', fontsize=14, fontweight='bold')
                plt.xlabel('AI Application')
                plt.ylabel('Suitability Score')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 100)
                plt.grid(True, alpha=0.3)
                
                # Add score labels on bars
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
                
                plot_path = os.path.join(output_dir, 'training_suitability_scores.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots['training_suitability_scores'] = plot_path
            
            # 5. Word cloud of most common words
            if WORDCLOUD_AVAILABLE and not self.df.empty:
                all_text = ' '.join(self.df['content'].astype(str))
                words = self.tokenize_text(all_text)
                words = [w for w in words if w.isalpha() and w not in self.stop_words and len(w) > 3]
                
                if words:
                    try:
                        wordcloud = WordCloud(width=800, height=400, 
                                            background_color='white',
                                            max_words=100,
                                            colormap='viridis').generate(' '.join(words))
                        
                        plt.figure(figsize=(12, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title('Most Common Words in Dataset', fontsize=14, fontweight='bold')
                        plot_path = os.path.join(output_dir, 'word_cloud.png')
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        plots['word_cloud'] = plot_path
                    except Exception as e:
                        print(f"Word cloud creation failed: {e}")
            
            print(f"Visualizations saved to {output_dir}/")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        return plots
    
    def print_summary_report(self):
        """Print a concise summary report to console"""
        report = self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("SCRAPED DATA ANALYSIS SUMMARY")
        print("="*80)
        
        # Dataset Overview
        overview = report['dataset_overview']
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Pages: {overview['total_pages']:,}")
        print(f"   Total Words: {overview['total_words']:,}")
        print(f"   Avg Words/Page: {overview['average_words_per_page']:.1f}")
        print(f"   Failed Scrapes: {overview['failed_scrapes']}")
        
        # Overall Assessment
        assessment = report['overall_assessment']
        if 'error' not in assessment:
            print(f"\n🎯 OVERALL ASSESSMENT:")
            print(f"   Quality Score: {assessment['overall_score']}/100 (Grade: {assessment['overall_grade']})")
            print(f"   Quality Tier: {assessment['quality_tier']}")
            print(f"   Best for: {assessment['best_application'].replace('_', ' ').title()} ({assessment['best_application_score']}/100)")
            print(f"   Description: {assessment['tier_description']}")
        
        # Training Suitability
        suitability = report['training_suitability']
        print(f"\n🤖 AI APPLICATION SUITABILITY:")
        for app, data in suitability.items():
            if 'score' in data:
                app_name = app.replace('_', ' ').title()
                score = data['score']
                grade = data['grade']
                print(f"   {app_name:20} {score:3.0f}/100 ({grade})")
        
        # Key Recommendations
        recommendations = report['recommendations']
        if recommendations:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)

# Main analysis function
def analyze_scraped_data(json_file: str, create_plots: bool = True, 
                        output_dir: str = "analysis_results") -> str:
    """
    Main function to analyze scraped data and generate comprehensive report
    
    Args:
        json_file: Path to the JSON file with scraped data
        create_plots: Whether to create visualization plots
        output_dir: Directory to save analysis results
    
    Returns:
        Path to the generated analysis report
    """
    import os
    
    print(f"Starting analysis of {json_file}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = ScrapedDataAnalyzer(json_file)
        
        # Print summary to console
        analyzer.print_summary_report()
        
        # Generate and save comprehensive report
        report_file = os.path.join(output_dir, 'comprehensive_analysis_report.json')
        analyzer.save_report(report_file)
        
        # Create visualizations if requested
        if create_plots:
            plot_dir = os.path.join(output_dir, 'plots')
            plots = analyzer.create_visualizations(plot_dir)
            print(f"Created {len(plots)} visualization plots")
        
        print(f"\n✅ Analysis complete! Results saved in: {output_dir}")
        return report_file
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis_results"
        
        try:
            report_file = analyze_scraped_data(json_file, create_plots=True, output_dir=output_dir)
            print(f"\n🎉 Analysis completed successfully!")
            print(f"Check {output_dir}/ for detailed results and visualizations.")
        except Exception as e:
            print(f"💥 Error: {e}")
            sys.exit(1)
    else:
        # Try to find scraped data files automatically
        json_files = glob.glob("*scraped_data*.json")
        if json_files:
            print(f"Found scraped data file: {json_files[0]}")
            try:
                report_file = analyze_scraped_data(
                    json_files[0],
                    create_plots=True,
                    output_dir="analysis_results"
                )
                print(f"\n🎉 Analysis completed successfully!")
            except Exception as e:
                print(f"Analysis failed: {e}")
        else:
            print("No scraped data files found.")
            print("Usage: python data_analyzer.py <path_to_scraped_data.json> [output_directory]")