import json
import gzip
import lz4.frame
import zstandard as zstd
import brotli
import lzma
import numpy as np
from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from typing import List, Callable, Optional

class ZIPFIT:
    def __init__(
        self, 
        source_dataset: str, 
        target_dataset: str,
        k: int,
        source_load_fn: Optional[Callable[[str], List[str]]] = None, 
        source_parse_fn: Optional[Callable[[dict], str]] = None,         
        target_load_fn: Optional[Callable[[str], List[str]]] = None, 
        target_parse_fn: Optional[Callable[[dict], str]] = None, 
        output_file: str = "top_k_sequences.jsonl",
        compression_algorithm: str = 'gzip',
        compress_level: int = 0
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_load_fn = source_load_fn
        self.source_parse_fn = source_parse_fn        
        self.target_load_fn = target_load_fn
        self.target_parse_fn = target_parse_fn
        self.output_file = output_file
        self.compress_cache = {}
        self.k = k
        self.compress_level = compress_level
        self.compression_algorithm = compression_algorithm
        
        """
        Initializes the ZIPFIT instance.

        Parameters:
        - source_dataset (str): Path to the source dataset or callable function.
        - target_dataset (str): Path to the target dataset or callable function.
        - source_load_fn (Callable): Function to load the source dataset.
        - source_parse_fn (Callable): Function to parse examples from the source dataset.
        - target_load_fn (Callable): Function to load the target dataset.
        - target_parse_fn (Callable): Function to parse examples from the target dataset.
        - k (int): Number of top sequences to output.
        - output_file (str): Name of the output file for the top K sequences (default: "top_k_sequences.jsonl").
        - compression_algorithm (str): Compression algorithm to use ('gzip' or 'lz4').
        """

    def load_jsonl(self, file_path: str) -> List[str]:
        """Loads and returns the text field from a JSONL file.

        Returns:
            List[str]: A list of text entries from the JSONL file.
        """
        data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    data.append(item['text']) 
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
        return data


    def load_and_process_datasets(self, file_path: Optional[str], load_fn: Optional[Callable], parse_fn: Optional[Callable]) -> list:
        """Loads and processes a dataset from a specified file path or function.

        Parameters:
            file_path (str): Path to the JSONL file (if applicable).
            load_fn (Callable): Function to load the dataset from HuggingFace(if applicable).
            parse_fn (Callable): Function to parse examples from the dataset from HuggingFace(if applicable).

        Returns:
            list: A list of processed sequences from the dataset.
        """
        if load_fn and parse_fn:
            raw_data = load_fn(file_path)  
            processed_data = [parse_fn(ex) for ex in raw_data]  
            return processed_data
        elif file_path and file_path.endswith('.jsonl'):
            return self.load_jsonl(file_path)  # Load from JSONL
        else:
            raise ValueError("Invalid input: Provide a JSONL file path or a valid HuggingFace loading/parsing function.")

    def compress(self, data: str) -> int:
        """Compresses the input data using the specified algorithm and returns the size of the compressed data, using a cache.

        Parameters:
            data (str): The input data to compress.

        Returns:
            int: The size of the compressed data.
        """
        if data in self.compress_cache:
            return self.compress_cache[data]
        
        if self.compression_algorithm == 'gzip':
            compressed_size = len(gzip.compress(data.encode('utf-8'), compresslevel=self.compress_level))
        elif self.compression_algorithm == 'lz4':
            compressed_size = len(lz4.frame.compress(data.encode('utf-8'), compression_level=self.compress_level))
        elif self.compression_algorithm == 'zstd':
            compressed_size = len(zstd.ZstdCompressor(level=self.compress_level).compress(data.encode('utf-8')))
        elif self.compression_algorithm == 'brotli':
            compressed_data = brotli.compress(data.encode('utf-8'))
        elif self.compression_algorithm == 'lzma':
            compressed_data = lzma.compress(data.encode('utf-8'))
        else:
            raise ValueError(f"Unsupported compression algorithm: {self.compression_algorithm}")

        self.compress_cache[data] = compressed_size  # Cache the result
        return compressed_size

    def normalized_compression_distance(self, c1: int, c2: int, c12: int) -> float:
        """Calculates the Normalized Compression Distance given precomputed compression sizes.

        Parameters:
            c1 (int): The compressed size of the first sequence.
            c2 (int): The compressed size of the second sequence.
            c12 (int): The compressed size of the concatenated sequences.

        Returns:
            float: The normalized compression distance.
        """
        return (c12 - min(c1, c2)) / max(c1, c2)

    def similarity(self, args) -> float:
        """Calculates similarity based on the normalized compression distance.

        Parameters:
            args (tuple): A tuple containing two sequences and their compressed sizes.

        Returns:
            float: The similarity score between the two sequences.
        """
        s1, s2, c1, c2 = args
        c12 = self.compress(s1 + s2)  # Use the cached compress function
        return 1 - self.normalized_compression_distance(c1, c2, c12)

    # Precompute GZIP sizes for each dataset
    def precompute_gzip_sizes(self, data):
        """Precompute the GZIP sizes for each sequence in the dataset.

        Parameters:
            data (list): A list of sequences to compress.

        Returns:
            list: A list of compressed sizes for the input sequences.
        """
        with Pool(cpu_count()) as pool:
            return pool.map(self.compress, data, chunksize = 2000)

    def rank_sequences_by_alignment(self, source_data: list, target_data: list) -> list:
        """Ranks sequences by alignment with reference data using dynamic scheduling.

        Parameters:
            source_data (list): A list of source data sequences.
            target_data (list): A list of target data sequences.

        Returns:
            list: A list of top K sequences with their alignment scores.
        """
        source_compressed = self.precompute_gzip_sizes(source_data)
        target_compressed = self.precompute_gzip_sizes(target_data)
        scores = []
        with Pool(cpu_count()) as pool:
            args_list = [(seq, ref, source_compressed[i], target_compressed[j]) 
                        for i, seq in enumerate(source_data) 
                        for j, ref in enumerate(target_data)]
            
            results = pool.imap_unordered(self.similarity, args_list, chunksize=2000)
            
            for result in results:
                scores.append(result)

        data_with_scores = list(zip(source_data, scores))
        top_k = sorted(data_with_scores, key=lambda x: x[1], reverse=True)[:self.k]
        return top_k
        
    def run(self):
        """Main execution function to run the data processing and scoring."""
        source_data = self.load_and_process_datasets(self.source_dataset, self.source_load_fn, self.source_parse_fn)
        target_data = self.load_and_process_datasets(self.target_dataset, self.target_load_fn, self.target_parse_fn) 
        top_k = self.rank_sequences_by_alignment(source_data, target_data)

        # Create a new JSONL file with the top K sequences
        with open(self.output_file, 'w') as f:
            for text, score in top_k:
                f.write(json.dumps({'text': text}) + '\n')

        print(f"Top {self.k} sequences saved to {self.output_file}")
    def main():
        # Define the paths and target dataset
        source_dataset = "/lfs/skampere1/0/eobbad/.cache/huggingface/hub/datasets--UDACA--Code-Mixed-Dataset/snapshots/4ea34026bf6a4b7cda65782406b7e32484a43ce0/combined_dataset.jsonl"
        target_dataset = 'openai/openai_humaneval'

        # Define the function to load the target dataset
        def target_load_dataset_fn(dataset):
            ds = load_dataset(dataset, split='test', trust_remote_code=True)
            return ds

        # Define the function to parse examples from the target dataset
        def target_parse_example_fn(ex):
            text = f"Problem description: {ex['prompt']} \nCanonical solution: {ex['canonical_solution']}"
            return text

        # Create an instance of ZIPFIT
        zip_fit_instance = ZIPFIT(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            target_load_fn=target_load_dataset_fn,
            target_parse_fn=target_parse_example_fn,
            k=100,  # Get top 10 sequences
            output_file="top_k_sequences.jsonl",
            compression_algorithm='gzip'  # Change to 'lz4' if desired
        )

        # Run the ZIPFIT process
        zip_fit_instance.run()

if __name__ == "__main__":
    main()


