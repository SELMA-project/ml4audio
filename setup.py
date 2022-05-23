import os
from setuptools import find_packages
from setuptools import setup


def req_file(filename, folder="./"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")

with open("README.md") as f:
    readme = f.read()


setup(
    name="ml4audio",
    version="0.1",
    author="Tilo Himmelsbach",
    author_email="dertilo@gmail.com",
    packages=find_packages(),
    license="MIT License",
    long_description=readme,
    install_requires=install_requires,
    python_requires=">=3.9",
)
