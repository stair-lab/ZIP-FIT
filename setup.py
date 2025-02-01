from setuptools import setup, find_packages

setup(
    # Basic package metadata
    name="zip-fit",
    version="1.0.8",
    author="Elyas Obbad",
    author_email="eobbad@stanford.edu",
    description="Data Selection via Compression-Based Alignment",
    
    # Use your README.md for the long description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    
    # Explicitly include the package "zip_fit" and any subpackages (e.g. zip_fit.submodule)
    # This ensures that find_packages() only returns packages under "zip_fit"
    packages=find_packages(include=["zip_fit", "zip_fit.*"]),
    
    # Specify the minimum Python version required
    python_requires=">=3.6",
    
    # List your runtime dependencies here. They will be installed when someone installs your package.
    install_requires=[
        "numpy>=1.21.0",
        "lz4>=3.1.10",
        "datasets>=1.17.0",
        "networkx>=2.5",
        "scipy",
        "scikit-learn",
        "pandas",
        "requests",
        "aiohttp",
        "matplotlib",
        "fire",
        "Levenshtein",
        "seaborn",
        "wandb",          
        "twine",
        "nvidia-htop",    
        "protobuf",
        "torch",
        "torchvision",
        "trl",
        "transformers",
        "accelerate>=0.26.0",
        "peft",
        "bitsandbytes",
        "einops",
        "sentencepiece",
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9",
            "black>=21.0",
            "mypy>=0.910",
        ],
    },
    
    # Classifiers help users (and tools) understand what your package is about.
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Developers",
    ],
    
    # Keywords for your package
    keywords="data selection compression zip-fit nlp language models",
    
    # This tells setuptools to include additional files as specified in MANIFEST.in
    include_package_data=True,
)
