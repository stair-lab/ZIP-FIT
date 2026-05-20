#!/usr/bin/env python3

from datasets import load_dataset
from collections import Counter

def main():
    # Load the combined dataset
    print("Loading dataset from zipfit/math-select-06062025...")
    dataset = load_dataset('zipfit/math-select-06062025')
    
    # Print basic information
    print(f"Splits: {list(dataset.keys())}")
    print(f"Size: {len(dataset['src'])} examples")
    print(f"Fields: {dataset['src'].features}")
    
    # Count examples by original source
    print("\nCounting examples by original source...")
    sources = Counter()
    
    for item in dataset['src']:
        sources[item['original_source']] += 1
    
    # Print source counts
    print("\nSource counts:")
    for source, count in sources.items():
        print(f"- {source}: {count}")
    
    # Verify total count matches expected
    print(f"\nTotal examples: {sum(sources.values())}")
    
    # Sample a few examples
    print("\nSample examples:")
    for i in range(min(3, len(dataset['src']))):
        item = dataset['src'][i]
        src = item['original_source']
        text_sample = item['text'][:200] + "..." if len(item['text']) > 200 else item['text']
        print(f"\nExample {i+1} from {src}:")
        print(text_sample)

if __name__ == "__main__":
    main() 