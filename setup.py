import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="single-cell-tracking-Nabil-Jabareen", # Replace with your own username
    version="0.0.1",
    author="Nabil Jabareen",
    author_email="njabareen@tum.de",
    description="Single cell detection and tracking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.lrz.de/single_cell_heterogeneity/models/single_cell_tracking",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
