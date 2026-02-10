from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="image_classification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An image classification project for learning and experimentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jurky123/image_classfication",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pillow>=8.3.0",
        "matplotlib>=3.4.0",
    ],
)
