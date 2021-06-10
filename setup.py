from setuptools import setup, find_packages

setup(
    name='uq360',
    version='0.1',
    url='https://github.com/IBM/UQ360',
    license='Apache License 2.0',
    author='uq360 developers',
    author_email='uq360@us.ibm.com',
    packages=find_packages(),
    description='IBM Uncertainty Quantification 360',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy>=1.16.5',
        'scipy>=1.2.0',
        'pandas>=0.24.0',
        'scikit-learn>=0.22',
        'matplotlib>=3.2',
        'autograd>=1.3',
        'torch>=1.6.0',
        'gpytorch>=1.3.0',
        'botorch>=0.3.2',
    ],
)
