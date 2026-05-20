import sys
sys.path.append('/lfs/skampere1/0/brando9/lm-evaluation-harness')

from datasets import load_dataset
from lm_eval.tasks.putnam_axiom.utils import process_variations

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("Putnam-AXIOM/putnam-axiom-dataset-ICML-2025-522")

# Check the variations split
variations = dataset["variations"]
print(f"Variations split size: {len(variations)}")

# After our fix, process_variations should filter for variation=1
filtered_variations = process_variations(variations)
print(f"Filtered variations size: {len(filtered_variations)}")

# Check if the filtering worked as expected
all_kept = len(filtered_variations) == len(variations)
print(f"All variations kept? {all_kept}")

# Sample a few examples to see what they look like
if len(filtered_variations) > 0:
    sample_index = 0
    print(f"\nSample problem: {filtered_variations[sample_index]['problem'][:100]}...")
    print(f"Variation value: {filtered_variations[sample_index]['variation']}") 