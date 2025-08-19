# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 15:16:02 2025

@author: Dr.PVVK
"""
import os
import sys

# Test if the analyzer can be imported
try:
    from data_analyzer import ScrapedDataAnalyzer
    print("✅ data_analyzer.py imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test if we can find scraped data
import glob
json_files = glob.glob("*scraped_data*.json")

if json_files:
    print(f"✅ Found data file: {json_files[0]}")
    try:
        analyzer = ScrapedDataAnalyzer(json_files[0])
        print("✅ Analyzer initialized successfully!")
        print(f"✅ Found {len(analyzer.df)} pages to analyze")
    except Exception as e:
        print(f"❌ Analyzer failed: {e}")
else:
    print("ℹ️  No scraped data files found - run the scraper first")

print("\n🎉 Setup appears to be working correctly!")