from setuptools import setup, find_packages

setup(
    name="hotspot_nmi",
    version="0.0.1",
    author="alouettevanhove",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "black",
        "pycodestyle",
        "pytest",
        "numpy",
        "scipy",
        "gymnasium",
        "properscoring",
        "matplotlib",
        "seaborn",
        "moviepy",
        'ipykernel',
    ],
)