from setuptools import setup, find_packages

setup(
    name="kolmogorov_machine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.7.0"],
        "tensorflow": ["tensorflow>=2.4.0"],
        "all": [
            "torch>=1.7.0",
            "tensorflow>=2.4.0",
            "scikit-learn>=0.24.0",
            "pandas>=1.1.0",
            "seaborn>=0.11.0",
        ],
    },
    author="Lemniscate World",
    author_email="example@example.com",
    description="A general-purpose neural network distributions analyzer and visualizer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lemniscate-world/Kolmogorov-Machine",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
