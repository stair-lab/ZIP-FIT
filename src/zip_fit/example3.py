# TODO: doesn't work yet
# from zip_fit import compute_zipfit_alignment
import sys
print(sys.path)
breakpoint()

def run_examples():
    # Example 1: Using the standalone function
    print("Example 1: Standalone function")
    texts_a = [
        "text",
    ]

    texts_b = [
        "text",
    ]

    alignment_score = compute_zipfit_alignment(texts_a, texts_b)
    print(f"Alignment score between text sets: {alignment_score}\n")

if __name__ == '__main__':
    run_examples()