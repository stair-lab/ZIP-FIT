from typing import List
from .zipfit import ZIPFIT

def compute_zipfit_alignment(texts_a: List[str], texts_b: List[str], compression_algorithm: str = 'lz4', compress_level: int = 0) -> float:
    """
    Compute the average compression-based alignment score between two sets of texts.
    
    This function measures how well two sets of texts align with each other using
    the ZIPFIT compression-based similarity metric. Higher scores indicate
    stronger alignment between the text sets.
    
    Args:
        texts_a: First collection of text strings
        texts_b: Second collection of text strings
        compression_algorithm: Algorithm to use ('gzip', 'lz4', 'zstd', 'brotli', or 'lzma')
        compress_level: Compression level to use
        
    Returns:
        float: Alignment score between 0-1, where 1 indicates perfect alignment
    """
    zipfit = ZIPFIT(None, None, k=1, compression_algorithm=compression_algorithm, 
                   compress_level=compress_level)
    return zipfit.compute_zipfit_alignment(texts_a, texts_b)

# Make these available at the package level
__all__ = ['ZIPFIT', 'compute_zipfit_alignment']