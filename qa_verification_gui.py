"""
Enhanced Dataset Debugger - Shows exact file structure and provides manual fix guidance
"""

import json
import os
from datetime import datetime

def deep_analyze_json(file_path):
    """Perform deep analysis of JSON structure"""
    print(f"Deep analysis of: {file_path}")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"File size: {os.path.getsize(file_path)} bytes")
        print(f"Root data type: {type(data).__name__}")
        
        def analyze_structure(obj, path="root", level=0):
            indent = "  " * level
            
            if isinstance(obj, dict):
                print(f"{indent}{path}: dict with {len(obj)} keys")
                for key, value in obj.items():
                    print(f"{indent}  Key: '{key}' -> {type(value).__name__}")
                    if isinstance(value, (dict, list)) and level < 3:
                        analyze_structure(value, f"{path}.{key}", level + 1)
                    elif isinstance(value, str) and len(value) > 100:
                        print(f"{indent}    Value preview: {value[:100]}...")
                    elif not isinstance(value, (dict, list)):
                        print(f"{indent}    Value: {value}")
            
            elif isinstance(obj, list):
                print(f"{indent}{path}: list with {len(obj)} items")
                if obj:
                    print(f"{indent}  First item type: {type(obj[0]).__name__}")
                    if level < 3:
                        analyze_structure(obj[0], f"{path}[0]", level + 1)
                    
                    if len(obj) > 1:
                        print(f"{indent}  Last item type: {type(obj[-1]).__name__}")
                        if type(obj[0]) != type(obj[-1]):
                            print(f"{indent}  WARNING: Mixed types in list!")
            
            else:
                print(f"{indent}{path}: {type(obj).__name__} = {obj}")
        
        analyze_structure(data)
        
        # Show raw structure preview
        print("\nRaw structure preview (first 500 chars):")
        print("-" * 50)
        raw_preview = json.dumps(data, indent=2)[:500]
        print(raw_preview)
        if len(json.dumps(data)) > 500:
            print("... (truncated)")
        
        return data
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def manual_fix_guidance(data):
    """Provide specific guidance on how to manually fix the dataset"""
    print("\n" + "=" * 60)
    print("MANUAL FIX GUIDANCE")
    print("=" * 60)
    
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Your file is a dictionary with keys: {keys}")
        
        # Check for common patterns
        if 'qa_pairs' in data:
            print("✓ Found 'qa_pairs' key - this looks promising")
            qa_pairs = data['qa_pairs']
            if isinstance(qa_pairs, list):
                print(f"✓ qa_pairs is a list with {len(qa_pairs)} items")
                if qa_pairs and isinstance(qa_pairs[0], dict):
                    print("✓ Items appear to be dictionaries")
                    sample_keys = list(qa_pairs[0].keys())
                    print(f"Sample keys: {sample_keys}")
                    
                    required_keys = ['question', 'answer']
                    missing_keys = [k for k in required_keys if k not in sample_keys]
                    if missing_keys:
                        print(f"❌ Missing required keys: {missing_keys}")
                    else:
                        print("✓ Has required keys (question, answer)")
                        print("This dataset should work! Try loading it again.")
                        return
                else:
                    print("❌ qa_pairs items are not dictionaries")
            else:
                print(f"❌ qa_pairs is not a list, it's: {type(qa_pairs)}")
        
        elif any(key in data for key in ['question', 'answer']):
            print("📝 This appears to be a single QA pair format")
            print("SOLUTION: Wrap it in the correct structure:")
            show_single_qa_fix(data)
            return
        
        else:
            print("🔍 Unknown dictionary format. Looking for QA-like data...")
            
            # Search for question/answer patterns
            qa_candidates = []
            for key, value in data.items():
                if isinstance(value, list):
                    for i, item in enumerate(value[:3]):  # Check first 3 items
                        if isinstance(item, dict):
                            item_keys = list(item.keys())
                            if any('question' in k.lower() for k in item_keys):
                                qa_candidates.append((key, i, item_keys))
            
            if qa_candidates:
                print("🎯 Found potential QA data:")
                for key, idx, item_keys in qa_candidates:
                    print(f"  {key}[{idx}]: {item_keys}")
                show_restructure_guidance(data, qa_candidates[0][0])
            else:
                print("❌ No obvious QA pattern found")
                show_custom_fix_guidance(data)
    
    elif isinstance(data, list):
        print(f"Your file is a list with {len(data)} items")
        if data and isinstance(data[0], dict):
            sample_keys = list(data[0].keys())
            print(f"Sample item keys: {sample_keys}")
            
            if 'question' in sample_keys and 'answer' in sample_keys:
                print("✓ This looks like a direct list of QA pairs")
                print("SOLUTION: Add metadata wrapper:")
                show_list_fix()
            else:
                print("❌ Items don't have question/answer keys")
                show_custom_fix_guidance(data)
        else:
            print("❌ List items are not dictionaries")
    else:
        print(f"❌ Unexpected root type: {type(data)}")

def show_single_qa_fix(data):
    """Show how to fix single QA pair format"""
    print("\nCREATE THIS STRUCTURE:")
    print("-" * 30)
    fixed_structure = {
        "metadata": {
            "total_pairs": 1,
            "source": "Manual fix"
        },
        "qa_pairs": [data]
    }
    print(json.dumps(fixed_structure, indent=2)[:300] + "...")

def show_list_fix():
    """Show how to fix direct list format"""
    print("\nWRAP YOUR LIST LIKE THIS:")
    print("-" * 30)
    print('''{
  "metadata": {
    "total_pairs": <NUMBER_OF_ITEMS>,
    "source": "Manual fix"
  },
  "qa_pairs": [
    // YOUR EXISTING LIST ITEMS GO HERE
  ]
}''')

def show_restructure_guidance(data, qa_key):
    """Show how to restructure when QA data is nested"""
    print(f"\nRESTRUCTURE GUIDANCE:")
    print("-" * 30)
    print(f"Move data from '{qa_key}' to 'qa_pairs':")
    print(f'''{{
  "metadata": {{
    "total_pairs": {len(data.get(qa_key, []))},
    "source": "Restructured from {qa_key}"
  }},
  "qa_pairs": [
    // Move items from data["{qa_key}"] here
  ]
}}''')

def show_custom_fix_guidance(data):
    """Show guidance for custom fixes"""
    print("\nCUSTOM FIX NEEDED:")
    print("-" * 30)
    print("Your data doesn't match standard QA patterns.")
    print("You need to manually create a structure like this:")
    print('''{
  "metadata": {
    "total_pairs": <NUMBER>,
    "source": "Manual conversion"
  },
  "qa_pairs": [
    {
      "question": "Your question text here",
      "answer": "Your answer text here",
      "context": "Optional context",
      "difficulty": "easy/medium/hard",
      "category": "category name",
      "source_page": "source information",
      "confidence": 0.8,
      "verified": false,
      "notes": "",
      "timestamp": ""
    }
  ]
}''')

def attempt_auto_fix(data, input_file):
    """Try harder to automatically fix the dataset"""
    print("\n" + "=" * 60)
    print("ATTEMPTING AUTO-FIX WITH ENHANCED LOGIC")
    print("=" * 60)
    
    fixed_data = None
    fix_method = ""
    
    if isinstance(data, dict):
        # Try different key variations
        qa_keys = ['qa_pairs', 'questions', 'data', 'pairs', 'items', 'qa_data']
        found_qa_key = None
        
        for key in qa_keys:
            if key in data and isinstance(data[key], list):
                found_qa_key = key
                break
        
        if found_qa_key:
            qa_list = data[found_qa_key]
            if qa_list and isinstance(qa_list[0], dict):
                # Check if items have question/answer or similar
                sample = qa_list[0]
                q_key = None
                a_key = None
                
                # Look for question/answer keys (case insensitive, partial match)
                for k in sample.keys():
                    k_lower = k.lower()
                    if 'question' in k_lower or 'q' == k_lower:
                        q_key = k
                    elif 'answer' in k_lower or 'a' == k_lower or 'response' in k_lower:
                        a_key = k
                
                if q_key and a_key:
                    # Standardize the data
                    standardized_pairs = []
                    for item in qa_list:
                        standardized_item = {
                            'question': str(item.get(q_key, '')),
                            'answer': str(item.get(a_key, '')),
                            'context': str(item.get('context', '')),
                            'difficulty': str(item.get('difficulty', 'medium')),
                            'category': str(item.get('category', 'general')),
                            'source_page': str(item.get('source_page', '')),
                            'confidence': float(item.get('confidence', 0.7)),
                            'verified': bool(item.get('verified', False)),
                            'notes': str(item.get('notes', '')),
                            'timestamp': str(item.get('timestamp', datetime.now().isoformat()))
                        }
                        if standardized_item['question'].strip() and standardized_item['answer'].strip():
                            standardized_pairs.append(standardized_item)
                    
                    if standardized_pairs:
                        fixed_data = {
                            'metadata': {
                                'total_pairs': len(standardized_pairs),
                                'source': f'Auto-fixed from {found_qa_key}',
                                'original_q_key': q_key,
                                'original_a_key': a_key,
                                'fixed_at': datetime.now().isoformat()
                            },
                            'qa_pairs': standardized_pairs
                        }
                        fix_method = f"Converted {found_qa_key} with keys '{q_key}'/'{a_key}' to standard format"
    
    elif isinstance(data, list) and data:
        # Try to fix list format
        if isinstance(data[0], dict):
            sample = data[0]
            q_key = None
            a_key = None
            
            for k in sample.keys():
                k_lower = k.lower()
                if 'question' in k_lower or 'q' == k_lower:
                    q_key = k
                elif 'answer' in k_lower or 'a' == k_lower or 'response' in k_lower:
                    a_key = k
            
            if q_key and a_key:
                standardized_pairs = []
                for item in data:
                    if isinstance(item, dict):
                        standardized_item = {
                            'question': str(item.get(q_key, '')),
                            'answer': str(item.get(a_key, '')),
                            'context': str(item.get('context', '')),
                            'difficulty': str(item.get('difficulty', 'medium')),
                            'category': str(item.get('category', 'general')),
                            'source_page': str(item.get('source_page', '')),
                            'confidence': float(item.get('confidence', 0.7)),
                            'verified': bool(item.get('verified', False)),
                            'notes': str(item.get('notes', '')),
                            'timestamp': str(item.get('timestamp', datetime.now().isoformat()))
                        }
                        if standardized_item['question'].strip() and standardized_item['answer'].strip():
                            standardized_pairs.append(standardized_item)
                
                if standardized_pairs:
                    fixed_data = {
                        'metadata': {
                            'total_pairs': len(standardized_pairs),
                            'source': 'Auto-fixed from list format',
                            'original_q_key': q_key,
                            'original_a_key': a_key,
                            'fixed_at': datetime.now().isoformat()
                        },
                        'qa_pairs': standardized_pairs
                    }
                    fix_method = f"Converted list with keys '{q_key}'/'{a_key}' to standard format"
    
    if fixed_data:
        # Save the fixed version
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_auto_fixed{ext}"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(fixed_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ SUCCESS: {fix_method}")
            print(f"✓ Saved {len(fixed_data['qa_pairs'])} QA pairs to: {output_file}")
            print(f"✓ You can now load this file in the GUI")
            return True
            
        except Exception as e:
            print(f"❌ Error saving fixed file: {e}")
            return False
    else:
        print("❌ Could not automatically fix this format")
        return False

def main():
    """Main function"""
    file_path = input("Enter the path to your JSON file (or press Enter for 'structured_qa_dataset.json'): ").strip()
    
    if not file_path:
        file_path = "structured_qa_dataset.json"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Deep analysis
    data = deep_analyze_json(file_path)
    
    if data is not None:
        # Try enhanced auto-fix first
        if not attempt_auto_fix(data, file_path):
            # If auto-fix fails, provide manual guidance
            manual_fix_guidance(data)

if __name__ == "__main__":
    main()