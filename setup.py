from setuptools import setup, find_packages

setup(
    name='cifar10-benchmark',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A benchmark repository for CIFAR-10 using various deep learning models.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'efficientnet-pytorch>=0.7.0',
        'pyyaml>=5.1',
        'tensorboard>=2.4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)