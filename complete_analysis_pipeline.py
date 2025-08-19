#!/usr/bin/env python3
"""
Complete Website Analysis Pipeline
This script runs the entire workflow: analyze → scrape → evaluate for AI training
"""

import os
import sys
import argparse
import glob
from datetime import datetime
from urllib.parse import urlparse

def run_complete_analysis_pipeline(website_url: str, 
                                 max_analysis_pages: int = 100,
                                 max_scrape_pages: int = None,
                                 output_base_dir: str = None):
    """
    Run the complete pipeline: website analysis → scraping → AI suitability evaluation
    
    Args:
        website_url: Target website URL
        max_analysis_pages: Maximum pages to analyze for discovery
        max_scrape_pages: Maximum pages to scrape (None = all)
        output_base_dir: Base directory for all outputs
    """
    
    print("🚀 COMPLETE WEBSITE ANALYSIS PIPELINE")
    print("="*80)
    print(f"🎯 Target: {website_url}")
    print(f"📊 Analysis Limit: {max_analysis_pages} pages")
    print(f"🔄 Scraping Limit: {'All discovered' if max_scrape_pages is None else max_scrape_pages} pages")
    
    # Setup directories
    domain = urlparse(website_url).netloc.replace('.', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if not output_base_dir:
        output_base_dir = f"{domain}_complete_analysis_{timestamp}"
    
    scraping_dir = os.path.join(output_base_dir, "scraped_data")
    analysis_dir = os.path.join(output_base_dir, "ai_analysis")
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(scraping_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    
    results = {
        'website_url': website_url,
        'timestamp': timestamp,
        'output_directory': output_base_dir,
        'steps_completed': [],
        'files_generated': []
    }
    
    try:
        # STEP 1: Website Analysis
        print(f"\n{'='*20} STEP 1: WEBSITE ANALYSIS {'='*20}")
        print("🔍 Discovering content structure and pages...")
        
        from website_analyzer import WebsiteContentAnalyzer
        
        analyzer = WebsiteContentAnalyzer(website_url, max_analysis_pages)
        config = analyzer.discover_content_locations()
        
        config_file = os.path.join(output_base_dir, 'scraper_config.json')
        analyzer.save_config(config_file)
        
        print(f"✅ Website analysis complete!")
        print(f"   📄 Pages discovered: {config['site_analysis']['total_pages_found']}")
        print(f"   ⭐ High-value pages: {config['site_analysis']['high_value_pages']}")
        print(f"   📁 Documents found: {config['site_analysis']['document_count']}")
        
        results['steps_completed'].append('website_analysis')
        results['files_generated'].append(config_file)
        results['analysis_stats'] = config['site_analysis']
        
        # STEP 2: Content Scraping
        print(f"\n{'='*20} STEP 2: CONTENT SCRAPING {'='*20}")
        print("📥 Extracting text content from all pages...")
        
        from website_scraper import WebsiteScraper
        
        scraper = WebsiteScraper(config_file)
        scraped_data = scraper.scrape_all_pages(max_scrape_pages)
        
        if not scraped_data:
            print("❌ No content was successfully scraped!")
            return results
        
        # Generate dataset files
        dataset_files = scraper.create_text_dataset(scraping_dir)
        
        print(f"✅ Content scraping complete!")
        print(f"   📊 Pages scraped: {len(scraped_data)}")
        print(f"   📝 Total words: {sum(content.word_count for content in scraped_data):,}")
        print(f"   💾 Files created: {len(dataset_files)}")
        
        results['steps_completed'].append('content_scraping')
        results['files_generated'].extend(dataset_files.values())
        results['scraping_stats'] = scraper.generate_statistics()
        
        # STEP 3: AI Training Suitability Analysis
        print(f"\n{'='*20} STEP 3: AI SUITABILITY ANALYSIS {'='*20}")
        print("🤖 Evaluating data quality for AI training applications...")
        
        from data_analyzer import ScrapedDataAnalyzer
        
        # Find the JSON data file
        json_files = glob.glob(os.path.join(scraping_dir, "*_scraped_data_*.json"))
        if not json_files:
            print("❌ Could not find scraped data JSON file!")
            return results
        
        json_file = json_files[0]  # Use the first (should be only) JSON file
        
        # Run comprehensive analysis
        data_analyzer = ScrapedDataAnalyzer(json_file)
        
        # Print summary to console
        data_analyzer.print_summary_report()
        
        # Generate detailed report and visualizations
        report_file = os.path.join(analysis_dir, 'ai_training_suitability_report.json')
        data_analyzer.save_report(report_file)
        
        # Create visualizations
        plot_dir = os.path.join(analysis_dir, 'visualizations')
        plots = data_analyzer.create_visualizations(plot_dir)
        
        print(f"✅ AI suitability analysis complete!")
        print(f"   📈 Report generated: {os.path.basename(report_file)}")
        print(f"   📊 Visualizations: {len(plots)} charts created")
        
        results['steps_completed'].append('ai_analysis')
        results['files_generated'].append(report_file)
        results['files_generated'].extend(plots.values())
        
        # Get overall assessment
        comprehensive_report = data_analyzer.generate_comprehensive_report()
        overall_assessment = comprehensive_report.get('overall_assessment', {})
        results['ai_assessment'] = overall_assessment
        
        # STEP 4: Generate Final Summary
        print(f"\n{'='*20} FINAL SUMMARY {'='*20}")
        
        final_summary = generate_final_summary(results, comprehensive_report)
        summary_file = os.path.join(output_base_dir, 'PIPELINE_SUMMARY.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(final_summary)
        
        results['files_generated'].append(summary_file)
        
        print(final_summary)
        
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 All results saved in: {output_base_dir}")
        print(f"📋 Summary report: PIPELINE_SUMMARY.txt")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required files are in the same directory:")
        print("- website_analyzer.py")
        print("- website_scraper.py") 
        print("- data_analyzer.py")
        return results
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        print(f"Completed steps: {results['steps_completed']}")
        return results

def generate_final_summary(results: dict, comprehensive_report: dict) -> str:
    """Generate a final summary report"""
    
    summary = f"""
{'='*80}
WEBSITE ANALYSIS PIPELINE - FINAL SUMMARY
{'='*80}

🎯 TARGET WEBSITE: {results['website_url']}
📅 ANALYSIS DATE: {results['timestamp']}
📁 OUTPUT DIRECTORY: {results['output_directory']}

{'='*80}
PIPELINE RESULTS
{'='*80}

✅ COMPLETED STEPS: {len(results['steps_completed'])}/3
   {'✅' if 'website_analysis' in results['steps_completed'] else '❌'} Website Analysis
   {'✅' if 'content_scraping' in results['steps_completed'] else '❌'} Content Scraping  
   {'✅' if 'ai_analysis' in results['steps_completed'] else '❌'} AI Suitability Analysis

📊 DATASET STATISTICS:
"""
    
    # Add analysis stats if available
    if 'analysis_stats' in results:
        stats = results['analysis_stats']
        summary += f"""   📄 Total Pages Discovered: {stats.get('total_pages_found', 'N/A')}
   ⭐ High-Value Pages: {stats.get('high_value_pages', 'N/A')}
   📁 Documents Found: {stats.get('document_count', 'N/A')}
"""
    
    # Add scraping stats if available
    if 'scraping_stats' in results:
        stats = results['scraping_stats']
        summary += f"""   📥 Pages Successfully Scraped: {stats.get('total_pages_scraped', 'N/A')}
   📝 Total Words Extracted: {stats.get('total_words', 'N/A'):,}
   📊 Average Words per Page: {stats.get('average_words_per_page', 'N/A')}
   💯 Success Rate: {stats.get('success_rate', 'N/A')}
"""
    
    # Add AI assessment if available
    if 'ai_assessment' in results and 'error' not in results['ai_assessment']:
        assessment = results['ai_assessment']
        summary += f"""
{'='*80}
AI TRAINING SUITABILITY ASSESSMENT
{'='*80}

🎯 OVERALL QUALITY SCORE: {assessment.get('overall_score', 'N/A')}/100 (Grade: {assessment.get('overall_grade', 'N/A')})
🏆 QUALITY TIER: {assessment.get('quality_tier', 'N/A')}
📋 DESCRIPTION: {assessment.get('tier_description', 'N/A')}

🥇 BEST APPLICATION: {assessment.get('best_application', 'N/A').replace('_', ' ').title()} 
   Score: {assessment.get('best_application_score', 'N/A')}/100

⚠️  WORST APPLICATION: {assessment.get('worst_application', 'N/A').replace('_', ' ').title()}
   Score: {assessment.get('worst_application_score', 'N/A')}/100

🤖 AI APPLICATION SCORES:
"""
        
        # Add individual application scores
        if 'score_distribution' in assessment:
            for app, score in assessment['score_distribution'].items():
                app_name = app.replace('_', ' ').title()
                grade = get_grade_from_score(score)
                summary += f"   {app_name:25} {score:3.0f}/100 ({grade})\n"
    
    # Add recommendations if available
    if comprehensive_report and 'recommendations' in comprehensive_report:
        recommendations = comprehensive_report['recommendations']
        summary += f"""
{'='*80}
KEY RECOMMENDATIONS
{'='*80}

"""
        for i, rec in enumerate(recommendations[:8], 1):
            summary += f"{i:2d}. {rec}\n"
    
    # Add file listing
    summary += f"""
{'='*80}
GENERATED FILES
{'='*80}

📁 OUTPUT STRUCTURE:
{results['output_directory']}/
├── 📄 scraper_config.json (Website analysis configuration)
├── 📄 PIPELINE_SUMMARY.txt (This summary report)
├── 📁 scraped_data/ (Raw scraped content)
│   ├── 📊 *.csv (Spreadsheet format dataset)
│   ├── 📋 *.json (Structured dataset with metadata)
│   ├── 📝 all_content.txt (Combined plain text)
│   └── 📈 scraping_statistics.json (Scraping metrics)
└── 📁 ai_analysis/ (AI training suitability analysis)
    ├── 📄 ai_training_suitability_report.json (Detailed analysis)
    └── 📁 visualizations/ (Charts and plots)
        ├── 📊 word_count_distribution.png
        ├── 📈 training_suitability_scores.png
        ├── 🥧 page_size_distribution.png
        ├── 📊 content_type_distribution.png
        └── ☁️  word_cloud.png

💾 TOTAL FILES GENERATED: {len(results['files_generated'])}

{'='*80}
NEXT STEPS
{'='*80}

Based on your data quality assessment, here are suggested next steps:

"""
    
    # Add next steps based on quality score
    if 'ai_assessment' in results and 'overall_score' in results['ai_assessment']:
        score = results['ai_assessment']['overall_score']
        best_app = results['ai_assessment'].get('best_application', '').replace('_', ' ')
        
        if score >= 75:
            summary += f"""✅ HIGH QUALITY DATA - Ready for AI Training!
   1. Start with {best_app} development (highest scoring application)
   2. Consider fine-tuning pre-trained models with your domain data
   3. Implement evaluation metrics for your specific use case
   4. Set up data versioning and quality monitoring
"""
        elif score >= 60:
            summary += f"""⚠️  GOOD QUALITY DATA - Some optimization recommended
   1. Focus on improving areas identified in the detailed report
   2. Consider data augmentation techniques for {best_app}
   3. Supplement with external datasets if needed
   4. Start with simpler models and gradually increase complexity
"""
        elif score >= 45:
            summary += f"""🔧 FAIR QUALITY DATA - Significant improvements needed
   1. Address the key issues identified in recommendations
   2. Consider expanding data collection to improve diversity
   3. Focus on data cleaning and preprocessing
   4. Start with basic models and prototype applications
"""
        else:
            summary += f"""🚨 POOR QUALITY DATA - Major improvements required
   1. Review and improve data collection strategy
   2. Focus on content quality over quantity
   3. Consider manual curation of high-quality examples
   4. Supplement heavily with external high-quality datasets
"""
    
    summary += f"""
{'='*80}
SUPPORT & DOCUMENTATION
{'='*80}

📖 For detailed analysis results, see:
   - ai_analysis/ai_training_suitability_report.json

📊 For data exploration, see:
   - ai_analysis/visualizations/ (charts and plots)

📋 For raw data access, see:
   - scraped_data/ (multiple formats available)

🔧 For technical details, see:
   - scraper_config.json (website analysis configuration)
   - scraping_statistics.json (scraping performance metrics)

{'='*80}
"""
    
    return summary

def get_grade_from_score(score: int) -> str:
    """Convert score to letter grade"""
    if score >= 90: return 'A+'
    elif score >= 85: return 'A'
    elif score >= 80: return 'A-'
    elif score >= 75: return 'B+'
    elif score >= 70: return 'B'
    elif score >= 65: return 'B-'
    elif score >= 60: return 'C+'
    elif score >= 55: return 'C'
    elif score >= 50: return 'C-'
    elif score >= 45: return 'D+'
    elif score >= 40: return 'D'
    else: return 'F'

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Complete Website Analysis Pipeline for AI Training Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complete_analysis_pipeline.py https://example.com
  python complete_analysis_pipeline.py https://example.com --analysis-pages 200 --scrape-pages 500
  python complete_analysis_pipeline.py https://example.com --output-dir my_analysis
        """
    )
    
    parser.add_argument('url', help='Website URL to analyze and scrape')
    parser.add_argument('--analysis-pages', type=int, default=100,
                       help='Max pages to analyze for discovery (default: 100)')
    parser.add_argument('--scrape-pages', type=int, default=None,
                       help='Max pages to scrape (default: all discovered)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            print("❌ Error: Invalid URL format")
            sys.exit(1)
    except:
        print("❌ Error: Invalid URL")
        sys.exit(1)
    
    # Check for required dependencies
    required_modules = ['requests', 'beautifulsoup4', 'pandas', 'nltk', 'sklearn', 'matplotlib', 'wordcloud']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required dependencies. Please install:")
        print(f"pip install {' '.join(missing_modules)}")
        sys.exit(1)
    
    # Run the complete pipeline
    try:
        results = run_complete_analysis_pipeline(
            website_url=args.url,
            max_analysis_pages=args.analysis_pages,
            max_scrape_pages=args.scrape_pages,
            output_base_dir=args.output_dir
        )
        
        if len(results['steps_completed']) == 3:
            print(f"\n🎉 SUCCESS! Complete analysis pipeline finished.")
            print(f"📁 Check your results in: {results['output_directory']}")
            sys.exit(0)
        else:
            print(f"\n⚠️  Pipeline partially completed. Check results in: {results['output_directory']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Example run with KL University
        print("🚀 Running example with KL University website...")
        print("💡 For custom analysis, run: python complete_analysis_pipeline.py <website_url>")
        
        try:
            results = run_complete_analysis_pipeline(
                website_url="https://kluniversity.in",
                max_analysis_pages=100,
                max_scrape_pages=None,
                output_base_dir="kl_university_complete_analysis"
            )
            
            if len(results['steps_completed']) == 3:
                print(f"\n🎉 Example completed successfully!")
                print(f"📁 Results saved in: {results['output_directory']}")
            else:
                print(f"⚠️  Example partially completed.")
                
        except Exception as e:
            print(f"💥 Example failed: {e}")
            print("\nPlease ensure all required dependencies are installed:")
            # print("pip install requests beautifulsoup4 pandas nltk scikit-learn matplotlib wordcloud textstat seaborn") '