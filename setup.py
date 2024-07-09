from setuptools import setup

import re

version: str = "1.0.5"

if not version:
    raise RuntimeError("Version is not set!")

setup(
    name="pysimilarities",
    author="alluding",
    version=version,
    url="https://github.com/alluding/compare",
    license="MIT",
    description="A Python wrapper for comparing texts for similarities.",
    install_requires=[
        "typing",
        "typing_extensions",
        "pydantic",
        "scikit-learn",
        "spacy",
        "numpy",
        "python-Levenshtein"
    ],
    python_requires=">=3.8.0",
)
