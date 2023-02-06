from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qamlz",
    version="1.2.1",
    description="Binary Classifier trained with D-Wave's Quantum Annealers.",
    packages=find_packages(include=["qamlz"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy >= 1.20.3",
        "scikit_learn >= 1.0.1",
        "scipy >= 1.7.1",
        "dwave-ocean-sdk >= 4.2.0",
    ],
    extras_require={
        "dev": [
            "pytest >= 3.7",
            "check-manifest >= 0.47",
        ],
    },
    url="https://github.com/tcoulvert/qaml-z",
    author="Thomas Sievert",
    author_email="63161166+tcoulvert@users.noreply.github.com",
)
