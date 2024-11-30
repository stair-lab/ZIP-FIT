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
from collections import defaultdict

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
        compress_level: int = 0,
        cache_size: int = 100000  # Add cache size limit
    ):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_load_fn = source_load_fn
        self.source_parse_fn = source_parse_fn        
        self.target_load_fn = target_load_fn
        self.target_parse_fn = target_parse_fn
        self.output_file = output_file
        self.compress_cache = {}
        self.cache_size = cache_size
        self.k = k
        self.compress_level = compress_level
        self.compression_algorithm = compression_algorithm

    def load_jsonl(self, file_path: str) -> List[str]:
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
        if load_fn and parse_fn:
            raw_data = load_fn(file_path)
            processed_data = [parse_fn(ex) for ex in raw_data]
            return processed_data
        elif file_path and file_path.endswith('.jsonl'):
            return self.load_jsonl(file_path)
        else:
            raise ValueError("Invalid input: Provide a JSONL file path or valid loading/parsing functions.")

    def compress(self, data: str) -> int:
        # Check cache first
        if data in self.compress_cache:
            return self.compress_cache[data]
        
        # Manage cache size
        if len(self.compress_cache) >= self.cache_size:
            # Clear half of the cache when it gets too large
            keys_to_remove = list(self.compress_cache.keys())[:(self.cache_size // 2)]
            for key in keys_to_remove:
                del self.compress_cache[key]

        # Compress based on selected algorithm
        if self.compression_algorithm == 'gzip':
            compressed_size = len(gzip.compress(data.encode('utf-8'), compresslevel=self.compress_level))
        elif self.compression_algorithm == 'lz4':
            compressed_size = len(lz4.frame.compress(data.encode('utf-8'), compression_level=self.compress_level))
        elif self.compression_algorithm == 'zstd':
            compressed_size = len(zstd.ZstdCompressor(level=self.compress_level).compress(data.encode('utf-8')))
        elif self.compression_algorithm == 'brotli':
            compressed_size = len(brotli.compress(data.encode('utf-8')))
        elif self.compression_algorithm == 'lzma':
            compressed_size = len(lzma.compress(data.encode('utf-8')))
        else:
            raise ValueError(f"Unsupported compression algorithm: {self.compression_algorithm}")

        self.compress_cache[data] = compressed_size
        return compressed_size

    def normalized_compression_distance(self, c1: int, c2: int, c12: int) -> float:
        return (c12 - min(c1, c2)) / max(c1, c2)

    def similarity(self, args) -> tuple:
        """Modified to return source index and similarity score"""
        s1, s2, c1, c2, source_idx = args
        c12 = self.compress(s1 + s2)
        similarity = 1 - self.normalized_compression_distance(c1, c2, c12)
        return (source_idx, similarity)

    def precompute_gzip_sizes(self, data):
        with Pool(cpu_count()) as pool:
            return pool.map(self.compress, data, chunksize=2000)

    def rank_sequences_by_alignment(self, source_data: list, target_data: list) -> list:
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

    def run(self):
        print("Loading datasets...")
        source_data = self.load_and_process_datasets(self.source_dataset, self.source_load_fn, self.source_parse_fn)
        target_data = self.load_and_process_datasets(self.target_dataset, self.target_load_fn, self.target_parse_fn)
        
        print(f"Loaded {len(source_data)} source sequences and {len(target_data)} target sequences")
        
        top_k = self.rank_sequences_by_alignment(source_data, target_data)
        
        print(f"Writing top {self.k} sequences to {self.output_file}")
        with open(self.output_file, 'w') as f:
            for text, score in top_k:
                f.write(json.dumps({'text': text}) + '\n')
