from setuptools import setup, find_packages

setup(
    name="zip-fit",
    version="1.0.8",
    author="Elyas Obbad",
    author_email="eobbad@stanford.edu",
    description="Data Selection via Compression-Based Alignment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # packages=find_packages(),
    
    package_dir={'': 'zip_fit'},
    packages=find_packages('zip_fit'),  # imports all modules/folders with  __init__.py & python files

    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.21.0",
        "lz4>=3.1.10",
        "datasets>=1.17.0",
        'networkx>=2.5',
        'scipy',
        'scikit-learn',
        'pandas',
        'requests',
        'aiohttp',
        'numpy',
        'matplotlib',
        'fire',
        'Levenshtein',
        'seaborn',
        'wandb',
        'twine',
        'nvidia-htop',
        'protobuf',

        'torch',
        'torchvision',
        # 'torchaudio',

        'trl',
        'transformers', 
        # 'accelerate',
        'accelerate>=0.26.0',
        'peft',

        'bitsandbytes',
        'einops',
        'sentencepiece',
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.9",
            "black>=21.0",
            "mypy>=0.910",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Linguistic",
        "Intended Audience :: Developers",
    ],
    keywords="data selection compression zip-fit nlp language models",
    include_package_data=True,
)
