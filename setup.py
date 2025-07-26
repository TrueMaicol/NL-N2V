from setuptools import setup, find_packages

setup(
    name="nl-n2v",
    version="1.0.0",
    description="Non-Local N2V: Improving N2V networks for spatially correlated noise",
    author="Diego Martin, Edoardo Peretti, Giacomo Boracchi",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1",
        "torchvision",
        "torchaudio", 
        "numpy>=2.2.0",
        "opencv-python>=4.10.0",
        "scikit-image",
        "pyyaml",
        "tensorboard",
        "matplotlib",
        "scipy",
        "pillow",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
