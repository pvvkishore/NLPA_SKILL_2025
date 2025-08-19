#!/usr/bin/env python3
"""
Complete Website Scraping Pipeline
This script combines the website analyzer and scraper to create a complete text dataset.
"""

import sys
import os
import argparse
from urllib.parse import urlparse

def run_complete_pipeline(website_url: str, max_analysis_pages: int = 100, 
                         max_scrape_pages: int = None, output_dir: str = None):
    """
    Run complete pipeline: analyze website, then scrape all content
    
    Args:
        website_url: URL of the website to scrape
        max_analysis_pages: Max pages to analyze for content discovery
        max_scrape_pages: Max pages to actually scrape (None = all discovered)
        output_dir: Output directory for dataset
    """
    
    print("=" * 80)
    print("WEBSITE SCRAPING PIPELINE")
    print("=" * 80)
    print(f"Target Website: {website_url}")
    print(f"Analysis Limit: {max_analysis_pages} pages")
    print(f"Scraping Limit: {'All discovered pages' if max_scrape_pages is None else f'{max_scrape_pages} pages'}")
    
    # Set default output directory
    if not output_dir:
        domain = urlparse(website_url).netloc.replace('.', '_')
        output_dir = f"{domain}_dataset"
    
    try:
        # Step 1: Import and run website analyzer
        print(f"\n{'='*20} STEP 1: ANALYZING WEBSITE {'='*20}")
        
        # Import the analyzer (assuming it's in the same directory or installed)
        from website_analyzer import WebsiteContentAnalyzer
        
        analyzer = WebsiteContentAnalyzer(website_url, max_analysis_pages)
        config = analyzer.discover_content_locations()
        
        # Save configuration
        config_file = 'scraper_config.json'
        analyzer.save_config(config_file)
        
        print(f"✓ Website analysis complete!")
        print(f"✓ Found {config['site_analysis']['total_pages_found']} pages")
        print(f"✓ Identified {config['site_analysis']['high_value_pages']} high-value pages")
        print(f"✓ Discovered {config['site_analysis']['document_count']} documents")
        
        # Step 2: Import and run website scraper
        print(f"\n{'='*20} STEP 2: SCRAPING CONTENT {'='*20}")
        
        # Import the scraper
        from website_scraper import WebsiteScraper
        
        scraper = WebsiteScraper(config_file)
        scraped_data = scraper.scrape_all_pages(max_scrape_pages)
        
        if not scraped_data:
            print("❌ No content was successfully scraped!")
            return None
        
        print(f"✓ Successfully scraped {len(scraped_data)} pages")
        
        # Step 3: Generate dataset
        print(f"\n{'='*20} STEP 3: CREATING DATASET {'='*20}")
        
        results = scraper.create_text_dataset(output_dir)
        
        if results:
            print(f"✓ Dataset creation complete!")
            print(f"✓ Files saved in: {results['output_directory']}")
            
            # Display final statistics
            stats = scraper.generate_statistics()
            print(f"\n{'='*20} FINAL STATISTICS {'='*20}")
            print(f"Total Pages Scraped: {stats['total_pages_scraped']}")
            print(f"Total Words Extracted: {stats['total_words']:,}")
            print(f"Average Words per Page: {stats['average_words_per_page']}")
            print(f"Success Rate: {stats['success_rate']}")
            print(f"Failed Pages: {stats['failed_pages']}")
            
            if stats['content_type_distribution']:
                print(f"\nContent Types Found:")
                for content_type, count in stats['content_type_distribution'].items():
                    print(f"  - {content_type}: {count} pages")
            
            print(f"\n{'='*20} OUTPUT FILES {'='*20}")
            print(f"📁 Dataset Directory: {results['output_directory']}")
            print(f"📄 JSON Dataset: {os.path.basename(results['json_file'])}")
            print(f"📊 CSV Dataset: {os.path.basename(results['csv_file'])}")
            print(f"📝 Text File: {os.path.basename(results['txt_file'])}")
            print(f"📈 Statistics: {os.path.basename(results['stats_file'])}")
            
            return results
        else:
            print("❌ Dataset creation failed!")
            return None
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure both website_analyzer.py and website_scraper.py are in the same directory.")
        return None
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        return None

def main():
    """Command line interface for the scraping pipeline"""
    parser = argparse.ArgumentParser(
        description="Complete Website Scraping Pipeline - Analyze and scrape any website",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scraping_pipeline.py https://example.com
  python scraping_pipeline.py https://example.com --analysis-pages 50 --scrape-pages 100
  python scraping_pipeline.py https://example.com --output-dir my_dataset
        """
    )
    
    parser.add_argument(
        'url',
        help='URL of the website to scrape (e.g., https://example.com)'
    )
    
    parser.add_argument(
        '--analysis-pages',
        type=int,
        default=100,
        help='Maximum pages to analyze for content discovery (default: 100)'
    )
    
    parser.add_argument(
        '--scrape-pages',
        type=int,
        default=None,
        help='Maximum pages to scrape (default: all discovered pages)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for dataset (default: domain_name_dataset)'
    )
    
    args = parser.parse_args()
    
    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            print("❌ Error: Invalid URL. Please provide a complete URL (e.g., https://example.com)")
            sys.exit(1)
    except Exception:
        print("❌ Error: Invalid URL format")
        sys.exit(1)
    
    # Run the pipeline
    results = run_complete_pipeline(
        website_url=args.url,
        max_analysis_pages=args.analysis_pages,
        max_scrape_pages=args.scrape_pages,
        output_dir=args.output_dir
    )
    
    if results:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Your dataset is ready in: {results['output_directory']}")
        sys.exit(0)
    else:
        print(f"\n💥 Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage for your KL University website
        print("Running example with KL University website...")
        
        results = run_complete_pipeline(
            website_url="https://kluniversity.in",
            max_analysis_pages=100,
            max_scrape_pages=None,  # Scrape all discovered pages
            output_dir="kl_university_dataset"
        )
        
        if results:
            print(f"\n🎉 Example completed successfully!")
            print(f"Check the '{results['output_directory']}' folder for your dataset.")
