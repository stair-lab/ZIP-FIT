"""
ZIPFIT: Embedding-Free Data Selection via Compression-Based Alignment

This module implements the ZIPFIT algorithm for selecting relevant training data
based on compression-based similarity metrics. ZIPFIT measures alignment between
datasets by leveraging the principle that similar texts compress better together.

Reference: https://arxiv.org/abs/2410.18194
"""

# Import necessary libraries for file operations and JSON handling
import json
# Import compression libraries for different compression algorithms
import gzip
import lz4.frame
import zstandard as zstd
import brotli
import lzma
# Import numpy for numerical operations
import numpy as np
# Import multiprocessing tools for parallel computation
from multiprocessing import Pool, cpu_count
# Import Hugging Face datasets library for dataset loading
from datasets import load_dataset
# Import typing for type annotations
from typing import List, Callable, Optional, Tuple, Dict, Any
# Import defaultdict for efficient dictionary operations
from collections import defaultdict


class ZIPFIT:
    """
    ZIPFIT: A compression-based data selection framework.
    
    This class implements the ZIPFIT algorithm for selecting the most relevant
    examples from a source dataset that align with a target dataset. The alignment
    is measured using compression-based similarity metrics, specifically the
    Normalized Compression Distance (NCD).
    
    Attributes:
        source_dataset (str): Path to source dataset or dataset identifier.
        target_dataset (str): Path to target dataset or dataset identifier.
        k (int): Number of top aligned examples to return.
        source_load_fn (Optional[Callable]): Function to load source dataset.
        source_parse_fn (Optional[Callable]): Function to extract text from source examples.
        target_load_fn (Optional[Callable]): Function to load target dataset.
        target_parse_fn (Optional[Callable]): Function to extract text from target examples.
        output_file (str): Path to output file for top-k aligned examples.
        compression_algorithm (str): Algorithm used for compression.
        compress_level (int): Compression level (0-9 for gzip).
        cache_size (int): Size of the compression cache.
    """

    def __init__(
        self, 
        source_dataset: str, 
        target_dataset: str,
        k: int,
        source_load_fn: Optional[Callable[[str], List[Any]]] = None, 
        source_parse_fn: Optional[Callable[[Any], str]] = None,         
        target_load_fn: Optional[Callable[[str], List[Any]]] = None, 
        target_parse_fn: Optional[Callable[[Any], str]] = None, 
        output_file: str = "top_k_sequences.jsonl",
        compression_algorithm: str = 'lz4',
        compress_level: int = 0,
        cache_size: int = 100000  # Cache size limit to prevent memory issues
    ):
        """
        Initialize the ZIPFIT instance.
        
        Args:
            source_dataset: Path to source dataset or dataset identifier.
            target_dataset: Path to target dataset or dataset identifier.
            k: Number of top aligned examples to return.
            source_load_fn: Function to load source dataset. If provided, should take a
                dataset path/identifier and return a list of examples.
            source_parse_fn: Function to extract text from source examples. If provided,
                should take an example and return a string.
            target_load_fn: Function to load target dataset. If provided, should take a
                dataset path/identifier and return a list of examples.
            target_parse_fn: Function to extract text from target examples. If provided,
                should take an example and return a string.
            output_file: Path to output file for top-k aligned examples.
            compression_algorithm: Algorithm used for compression ('gzip', 'lz4', 'zstd',
                'brotli', or 'lzma').
            compress_level: Compression level (0-9 for gzip).
            cache_size: Size of the compression cache to avoid redundant calculations.
        """
        # Store the path or identifier for the source dataset
        self.source_dataset = source_dataset
        # Store the path or identifier for the target dataset
        self.target_dataset = target_dataset
        # Store the custom function for loading the source dataset
        self.source_load_fn = source_load_fn
        # Store the custom function for parsing source examples
        self.source_parse_fn = source_parse_fn        
        # Store the custom function for loading the target dataset
        self.target_load_fn = target_load_fn
        # Store the custom function for parsing target examples
        self.target_parse_fn = target_parse_fn
        # Store the path for the output file
        self.output_file = output_file
        # Initialize an empty dictionary to cache compression results
        self.compress_cache: Dict[str, int] = {}
        # Store the maximum size for the compression cache
        self.cache_size = cache_size
        # Store the number of top examples to return
        self.k = k
        # Store the compression level for the selected algorithm
        self.compress_level = compress_level
        # Store the selected compression algorithm
        self.compression_algorithm = compression_algorithm

    def load_jsonl(self, file_path: str) -> List[str]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file.
            
        Returns:
            List of text strings extracted from the 'text' field of each JSON object.
            
        Raises:
            FileNotFoundError: If the file is not found.
            json.JSONDecodeError: If the file contains invalid JSON.
            """
        # Initialize an empty list to store the loaded data
        data = []
        try:
            # Open the file for reading
            with open(file_path, 'r') as f:
                # Process each line in the file
                for line in f:
                    # Parse the JSON object from the line
                    item = json.loads(line)
                    # Extract the 'text' field and add it to the data list
                    data.append(item['text']) 
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError:
            # Handle the case where the file contains invalid JSON
            print("Error: Failed to decode JSON from the file.")
        # Return the list of extracted text strings
        return data

    def load_and_process_datasets(self, file_path: Optional[str], 
                                 load_fn: Optional[Callable], 
                                 parse_fn: Optional[Callable]) -> List[str]:
        """
        Load and process datasets using custom functions or default JSONL loading.
        
        Args:
            file_path: Path to the dataset file or dataset identifier.
            load_fn: Function to load the dataset. If provided, should take a
                dataset path/identifier and return a list of examples.
            parse_fn: Function to extract text from examples. If provided,
                should take an example and return a string.
                
        Returns:
            List of processed text strings.
            
        Raises:
            ValueError: If neither valid loading/parsing functions nor a JSONL file path is provided.
        """
        # Check if both custom loading and parsing functions are provided
        if load_fn and parse_fn:
            # Use the custom loading function to load the raw data
            raw_data = load_fn(file_path)
            # Use the custom parsing function to extract text from each example
            processed_data = [parse_fn(ex) for ex in raw_data]
            # Return the processed data
            return processed_data
        # Check if a JSONL file path is provided
        elif file_path and file_path.endswith('.jsonl'):
            # Use the default JSONL loading method
            return self.load_jsonl(file_path)
        else:
            # Raise an error if neither valid functions nor a JSONL file path is provided
            raise ValueError("Invalid input: Provide a JSONL file path or valid loading/parsing functions.")

    def compress(self, data: str) -> int:
        """
        Compress a string and return its compressed size.
        
        This function uses the specified compression algorithm to compress the input
        string and returns the size of the compressed data. It uses caching to avoid
        redundant compression of the same string.
        
        Args:
            data: The string to compress.
            
        Returns:
            The size of the compressed data in bytes.
            
        Raises:
            ValueError: If an unsupported compression algorithm is specified.
        """
        # Check if the data is already in the cache to avoid redundant compression
        if data in self.compress_cache:
            # Return the cached compressed size
            return self.compress_cache[data]
        
        # Check if the cache has reached its size limit
        if len(self.compress_cache) >= self.cache_size:
            # Get the first half of the cache keys to remove
            keys_to_remove = list(self.compress_cache.keys())[:(self.cache_size // 2)]
            # Remove the selected keys from the cache
            for key in keys_to_remove:
                del self.compress_cache[key]

        # Compress the data using the gzip algorithm
        if self.compression_algorithm == 'gzip':
            # Encode the string to bytes, compress it, and get the length
            compressed_size = len(gzip.compress(data.encode('utf-8'), compresslevel=self.compress_level))
        # Compress the data using the lz4 algorithm
        elif self.compression_algorithm == 'lz4':
            # Encode the string to bytes, compress it, and get the length
            compressed_size = len(lz4.frame.compress(data.encode('utf-8'), compression_level=self.compress_level))
        # Compress the data using the zstd algorithm
        elif self.compression_algorithm == 'zstd':
            # Encode the string to bytes, compress it, and get the length
            compressed_size = len(zstd.ZstdCompressor(level=self.compress_level).compress(data.encode('utf-8')))
        # Compress the data using the brotli algorithm
        elif self.compression_algorithm == 'brotli':
            # Encode the string to bytes, compress it, and get the length
            compressed_size = len(brotli.compress(data.encode('utf-8')))
        # Compress the data using the lzma algorithm
        elif self.compression_algorithm == 'lzma':
            # Encode the string to bytes, compress it, and get the length
            compressed_size = len(lzma.compress(data.encode('utf-8')))
        else:
            # Raise an error for unsupported compression algorithms
            raise ValueError(f"Unsupported compression algorithm: {self.compression_algorithm}")

        # Cache the compressed size for future use
        self.compress_cache[data] = compressed_size
        # Return the compressed size
        return compressed_size

    def normalized_compression_distance(self, c1: int, c2: int, c12: int) -> float:
        """
        Calculate the Normalized Compression Distance (NCD) between two strings.
        
        NCD is defined as (C(xy) - min(C(x), C(y))) / max(C(x), C(y)), where:
        - C(x) is the compressed size of string x
        - C(y) is the compressed size of string y
        - C(xy) is the compressed size of the concatenation of x and y
        
        Args:
            c1: Compressed size of the first string.
            c2: Compressed size of the second string.
            c12: Compressed size of the concatenation of both strings.
            
        Returns:
            The NCD value, which ranges from 0 (identical) to 1 (completely different).
        """
        # Calculate the Normalized Compression Distance using the formula
        return (c12 - min(c1, c2)) / max(c1, c2)

    def similarity(self, comparison_inputs: Tuple[str, str, int, int, int]) -> Tuple[int, float]:
        """
        Calculate the similarity between two strings based on NCD.
        
        The similarity is defined as 1 - NCD, so it ranges from 0 (completely different)
        to 1 (identical).
        
        Args:
            comparison_inputs: Tuple containing (source_string, target_string, source_compressed_size,
                  target_compressed_size, source_index).
                  
        Returns:
            Tuple of (source_index, similarity_score).
        """
        # Unpack the arguments tuple
        source_text, target_text, source_compressed_size, target_compressed_size, source_idx = comparison_inputs
        # Compress the concatenation of the two strings
        combined_compressed_size = self.compress(source_text + target_text)
        # Calculate the similarity as 1 minus the normalized compression distance
        similarity_score = 1 - self.normalized_compression_distance(source_compressed_size, target_compressed_size, combined_compressed_size)
        # Return the source index and the calculated similarity
        return (source_idx, similarity_score)

    def precompute_gzip_sizes(self, text_data: List[str]) -> List[int]:
        """
        Precompute compression sizes for a list of strings in parallel.
        
        Args:
            text_data: List of strings to compress.
            
        Returns:
            List of compressed sizes corresponding to each input string.
        """
        # Create a multiprocessing pool with the number of available CPU cores
        with Pool(cpu_count()) as pool:
            # Map the compress function to each string in the data list
            return pool.map(self.compress, text_data, chunksize=2000)

    def rank_sequences_by_alignment(self, source_data: list, target_data: list) -> list:
        """
        Rank source sequences by their alignment with the target dataset.
        
        This function computes the average similarity between each source sequence
        and all target sequences, then returns the top-k most aligned source sequences.
        
        Args:
            source_data: List of source sequences.
            target_data: List of target sequences.
            
        Returns:
            List of tuples (source_sequence, alignment_score) for the top-k most aligned sequences.
        """
        print("Precomputing compression sizes...")
        source_compressed = self.precompute_gzip_sizes(source_data)
        target_compressed = self.precompute_gzip_sizes(target_data)
        
        print("Computing similarities...")
        # Create arguments list with source indices
        args_list = [
            (seq, ref, source_compressed[i], target_compressed[j], i) 
            for i, seq in enumerate(source_data) 
            for j, ref in enumerate(target_data)
        ]
        
        # Calculate similarities using multiprocessing
        similarities_by_source = defaultdict(list)
        with Pool(cpu_count()) as pool:
            for source_idx, similarity in pool.imap_unordered(self.similarity, args_list, chunksize=2000):
                similarities_by_source[source_idx].append(similarity)
        
        # Average similarities for each source sequence
        avg_similarities = [
            (source_data[idx], np.mean(scores))
            for idx, scores in similarities_by_source.items()
        ]
        
        # Sort by average similarity and take top k
        top_k = sorted(avg_similarities, key=lambda x: x[1], reverse=True)[:self.k]
        return top_k
    
    def compute_zipfit_alignment(self, texts_a: List[str], texts_b: List[str]) -> float:
        """
        Compute the average compression-based alignment score between two sets of texts.
        
        This function measures how well two sets of texts align with each other using
        the ZIPFIT compression-based similarity metric. Higher scores indicate
        stronger alignment between the text sets.
        
        Args:
            texts_a: First collection of text strings
            texts_b: Second collection of text strings
            
        Returns:
            float: Alignment score between 0-1, where 1 indicates perfect alignment
        """
        # Validate inputs
        if not texts_a or not texts_b:
            return 0.0
        
        print(f"Computing compression sizes for {len(texts_a)} and {len(texts_b)} texts...")
        # Directly compute compression sizes without multiprocessing
        compressed_a = [self.compress(text) for text in texts_a]
        compressed_b = [self.compress(text) for text in texts_b]
        
        print("Computing alignment scores...")
        # Calculate similarities directly
        total_alignment = 0.0
        count = 0
        
        for i, text_a in enumerate(texts_a):
            for j, text_b in enumerate(texts_b):
                # Create args tuple similar to the one passed to self.similarity
                args = (text_a, text_b, compressed_a[i], compressed_b[j], i)
                _, similarity = self.similarity(args)
                total_alignment += similarity
                count += 1
        
        # Return the average alignment score
        if count > 0:
            avg_alignment = total_alignment / count
            print(f"ZIPFIT alignment score: {avg_alignment:.4f}")
            return avg_alignment
        else:
            return 0.0

    def run(self) -> None:
        """
        Run the complete ZIPFIT process.
        
        This function:
        1. Loads and processes the source and target datasets
        2. Ranks source sequences by their alignment with the target dataset
        3. Writes the top-k most aligned sequences to the output file
        
        Returns:
            None
        """
        # Log the start of the dataset loading process
        print("Loading datasets...")
        # Load and process the source dataset
        source_data = self.load_and_process_datasets(self.source_dataset, self.source_load_fn, self.source_parse_fn)
        # Load and process the target dataset
        target_data = self.load_and_process_datasets(self.target_dataset, self.target_load_fn, self.target_parse_fn)
        
        # Log the number of loaded sequences
        print(f"Loaded {len(source_data)} source sequences and {len(target_data)} target sequences")
        
        # Rank the source sequences by their alignment with the target dataset
        top_k_aligned_sequences = self.rank_sequences_by_alignment(source_data, target_data)
        
        # Log the start of the output writing process
        print(f"Writing top {self.k} sequences to {self.output_file}")
        # Open the output file for writing
        with open(self.output_file, 'w') as f:
            # Write each top sequence to the output file as a JSON object
            for text, alignment_score in top_k_aligned_sequences:
                f.write(json.dumps({'text': text}) + '\n')

        
