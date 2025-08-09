import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import json
import re
from typing import Dict, List, Set, Tuple
import time
from collections import defaultdict
import mimetypes

class WebsiteContentAnalyzer:
    def __init__(self, base_url: str, max_pages: int = 100):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.max_pages = max_pages
        self.visited_urls = set()
        self.content_locations = defaultdict(list)
        self.document_links = []
        self.content_patterns = {}
        self.site_structure = {}
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def discover_content_locations(self) -> Dict:
        """Main method to analyze website and discover all content locations"""
        print(f"Starting analysis of {self.base_url}")
        
        # Step 1: Discover all pages and structure
        self._crawl_sitemap()
        self._discover_pages()
        
        # Step 2: Analyze content patterns
        self._analyze_content_patterns()
        
        # Step 3: Find document links
        self._find_document_links()
        
        # Step 4: Analyze site structure
        self._analyze_site_structure()
        
        # Step 5: Generate scraper configuration
        return self._generate_scraper_config()
    
    def _crawl_sitemap(self):
        """Try to find and parse sitemap.xml"""
        sitemap_urls = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml",
            f"{self.base_url}/robots.txt"
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    if 'sitemap.xml' in sitemap_url:
                        self._parse_sitemap(response.text)
                    elif 'robots.txt' in sitemap_url:
                        self._parse_robots_txt(response.text)
            except Exception as e:
                print(f"Could not access {sitemap_url}: {e}")
    
    def _parse_sitemap(self, sitemap_content: str):
        """Parse sitemap XML to find all URLs"""
        try:
            soup = BeautifulSoup(sitemap_content, 'xml')
            urls = soup.find_all('loc')
            for url in urls:
                if url.text and self.domain in url.text:
                    self.content_locations['sitemap_urls'].append(url.text)
        except Exception as e:
            print(f"Error parsing sitemap: {e}")
    
    def _parse_robots_txt(self, robots_content: str):
        """Parse robots.txt to find sitemap references and allowed paths"""
        lines = robots_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                try:
                    response = requests.get(sitemap_url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        self._parse_sitemap(response.text)
                except:
                    pass
    
    def _discover_pages(self):
        """Crawl the website to discover all accessible pages"""
        to_visit = [self.base_url]
        visited_count = 0
        
        while to_visit and visited_count < self.max_pages:
            current_url = to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            try:
                response = requests.get(current_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    self.visited_urls.add(current_url)
                    visited_count += 1
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find all internal links
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        full_url = urljoin(current_url, href)
                        
                        # Only follow internal links
                        if self.domain in full_url and full_url not in self.visited_urls:
                            to_visit.append(full_url)
                    
                    # Analyze this page's content
                    self._analyze_page_content(current_url, soup)
                    
                    time.sleep(0.5)  # Be respectful
                    
            except Exception as e:
                print(f"Error accessing {current_url}: {e}")
    
    def _analyze_page_content(self, url: str, soup: BeautifulSoup):
        """Analyze individual page content structure"""
        page_info = {
            'url': url,
            'title': '',
            'content_areas': [],
            'document_links': [],
            'content_types': [],
            'metadata': {}
        }
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            page_info['title'] = title_tag.get_text().strip()
        
        # Identify main content areas
        content_selectors = [
            ('main_content', ['main', 'article', '.content', '.post', '.entry']),
            ('navigation', ['nav', '.navigation', '.menu', '.nav']),
            ('sidebar', ['.sidebar', '.widget', 'aside']),
            ('header', ['header', '.header']),
            ('footer', ['footer', '.footer'])
        ]
        
        for area_type, selectors in content_selectors:
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text_content = element.get_text().strip()
                    if len(text_content) > 50:  # Only substantial content
                        page_info['content_areas'].append({
                            'type': area_type,
                            'selector': selector,
                            'text_length': len(text_content),
                            'html_structure': str(element)[:200] + '...'
                        })
        
        # Find content patterns
        self._identify_content_patterns(soup, page_info)
        
        # Find document links on this page
        self._find_page_documents(soup, url, page_info)
        
        # Store page analysis
        self.content_locations['pages'].append(page_info)
    
    def _identify_content_patterns(self, soup: BeautifulSoup, page_info: Dict):
        """Identify common content patterns and structures"""
        patterns = {
            'headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            'paragraphs': len(soup.find_all('p')),
            'lists': len(soup.find_all(['ul', 'ol'])),
            'tables': len(soup.find_all('table')),
            'forms': len(soup.find_all('form')),
            'images': len(soup.find_all('img')),
            'videos': len(soup.find_all(['video', 'iframe']))
        }
        
        # Check for specific content types
        content_indicators = {
            'blog_post': bool(soup.find_all(class_=re.compile(r'post|article|blog', re.I))),
            'product_page': bool(soup.find_all(class_=re.compile(r'product|item|catalog', re.I))),
            'documentation': bool(soup.find_all(class_=re.compile(r'doc|guide|manual', re.I))),
            'faq': bool(soup.find_all(class_=re.compile(r'faq|question|answer', re.I))),
            'forum': bool(soup.find_all(class_=re.compile(r'forum|discussion|thread', re.I)))
        }
        
        page_info['content_patterns'] = patterns
        page_info['content_types'] = [k for k, v in content_indicators.items() if v]
    
    def _find_page_documents(self, soup: BeautifulSoup, base_url: str, page_info: Dict):
        """Find downloadable documents on this page"""
        document_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.csv']
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if it's a document
            for ext in document_extensions:
                if ext in href.lower():
                    doc_info = {
                        'url': full_url,
                        'type': ext,
                        'title': link.get_text().strip(),
                        'context': str(link.parent)[:100] + '...' if link.parent else ''
                    }
                    page_info['document_links'].append(doc_info)
                    self.document_links.append(doc_info)
    
    def _find_document_links(self):
        """Comprehensive document discovery across the site"""
        common_doc_paths = [
            '/downloads/', '/docs/', '/documentation/', '/resources/', 
            '/files/', '/assets/', '/media/', '/papers/', '/reports/'
        ]
        
        for path in common_doc_paths:
            try:
                test_url = f"{self.base_url}{path}"
                response = requests.get(test_url, headers=self.headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    self._find_page_documents(soup, test_url, {'document_links': []})
            except:
                continue
    
    def _analyze_content_patterns(self):
        """Analyze overall content patterns across the site"""
        all_patterns = defaultdict(list)
        
        for page in self.content_locations['pages']:
            for pattern_type, count in page.get('content_patterns', {}).items():
                all_patterns[pattern_type].append(count)
        
        # Calculate averages and identify high-content pages
        pattern_summary = {}
        for pattern_type, counts in all_patterns.items():
            if counts:
                pattern_summary[pattern_type] = {
                    'average': sum(counts) / len(counts),
                    'max': max(counts),
                    'total': sum(counts)
                }
        
        self.content_patterns = pattern_summary
    
    def _analyze_site_structure(self):
        """Analyze overall site structure and organization"""
        url_patterns = defaultdict(list)
        content_types = defaultdict(int)
        
        for page in self.content_locations['pages']:
            # Analyze URL patterns
            parsed_url = urlparse(page['url'])
            path_parts = [part for part in parsed_url.path.split('/') if part]
            
            if path_parts:
                url_patterns[path_parts[0]].append(page['url'])
            
            # Count content types
            for content_type in page.get('content_types', []):
                content_types[content_type] += 1
        
        self.site_structure = {
            'url_patterns': dict(url_patterns),
            'content_type_distribution': dict(content_types),
            'total_pages': len(self.content_locations['pages']),
            'total_documents': len(self.document_links)
        }
    
    def _generate_scraper_config(self) -> Dict:
        """Generate comprehensive configuration for the scraper"""
        
        # Prioritize pages by content richness
        high_value_pages = []
        for page in self.content_locations['pages']:
            content_score = 0
            patterns = page.get('content_patterns', {})
            
            # Score based on content richness
            content_score += patterns.get('paragraphs', 0) * 2
            content_score += patterns.get('headings', 0) * 3
            content_score += patterns.get('lists', 0) * 1
            content_score += len(page.get('content_types', [])) * 5
            
            if content_score > 10:  # Threshold for valuable content
                high_value_pages.append({
                    'url': page['url'],
                    'title': page['title'],
                    'score': content_score,
                    'content_types': page.get('content_types', []),
                    'selectors': self._generate_content_selectors(page)
                })
        
        # Sort by content score
        high_value_pages.sort(key=lambda x: x['score'], reverse=True)
        
        # Generate scraper configuration
        scraper_config = {
            'base_url': self.base_url,
            'domain': self.domain,
            'site_analysis': {
                'total_pages_found': len(self.content_locations['pages']),
                'high_value_pages': len(high_value_pages),
                'document_count': len(self.document_links),
                'content_patterns': self.content_patterns,
                'site_structure': self.site_structure
            },
            'scraping_targets': {
                'high_priority_pages': high_value_pages[:50],  # Top 50 pages
                'document_links': self.document_links,
                'all_discovered_urls': [page['url'] for page in self.content_locations['pages']]
            },
            'content_extraction_rules': {
                'primary_content_selectors': [
                    'main', 'article', '.content', '.post', '.entry-content',
                    '.article-body', '.post-content', '.page-content'
                ],
                'text_selectors': ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'],
                'exclude_selectors': [
                    'nav', 'footer', '.sidebar', '.advertisement', '.ads',
                    '.menu', '.navigation', 'header', '.header'
                ]
            },
            'document_extraction': {
                'pdf_links': [doc for doc in self.document_links if '.pdf' in doc['type']],
                'office_docs': [doc for doc in self.document_links if doc['type'] in ['.doc', '.docx', '.ppt', '.pptx']],
                'data_files': [doc for doc in self.document_links if doc['type'] in ['.csv', '.txt', '.json']]
            },
            'scraping_settings': {
                'delay_between_requests': 1,
                'max_concurrent_requests': 3,
                'timeout': 30,
                'retry_attempts': 3,
                'respect_robots_txt': True
            }
        }
        
        return scraper_config
    
    def _generate_content_selectors(self, page_info: Dict) -> List[str]:
        """Generate specific content selectors for this page"""
        selectors = []
        
        for content_area in page_info.get('content_areas', []):
            if content_area['type'] == 'main_content' and content_area['text_length'] > 200:
                selectors.append(content_area['selector'])
        
        return selectors if selectors else ['main', 'article', '.content']
    
    def save_config(self, filename: str = 'scraper_config.json'):
        """Save the scraper configuration to a JSON file"""
        config = self.discover_content_locations()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Scraper configuration saved to {filename}")
        return config

# Usage example
def analyze_website(url: str, max_pages: int = 100) -> Dict:
    """
    Main function to analyze a website and generate scraper configuration
    
    Args:
        url: The base URL of the website to analyze
        max_pages: Maximum number of pages to crawl for analysis
    
    Returns:
        Dictionary containing complete scraper configuration
    """
    analyzer = WebsiteContentAnalyzer(url, max_pages)
    config = analyzer.save_config()
    
    # Print summary
    print(f"\n=== Website Analysis Complete ===")
    print(f"Website: {url}")
    print(f"Pages analyzed: {config['site_analysis']['total_pages_found']}")
    print(f"High-value pages: {config['site_analysis']['high_value_pages']}")
    print(f"Documents found: {config['site_analysis']['document_count']}")
    print(f"Content types discovered: {list(config['site_analysis']['site_structure']['content_type_distribution'].keys())}")
    
    return config

# Example usage:
if __name__ == "__main__":
    # Example: Analyze a website
    website_url = "https://kluniversity.in"  # Replace with target website
    config = analyze_website(website_url, max_pages=100)
