from setuptools import setup, find_packages
import re


# parameter variables
install_requires = []
dependency_links = []
package_data = {}
# determine requirements
with open('requirements.txt') as f:
    requirements = f.read()
for line in re.split('\n', requirements):
    if line and line[0] != '#' and '# test' not in line:
        install_requires.append(line)

setup(
    name='uq360',
    version='0.2',
    url='https://github.com/IBM/UQ360',
    license='Apache License 2.0',
    author='uq360 developers',
    author_email='uq360@us.ibm.com',
    packages=find_packages(),
    description='Uncertainty Quantification 360',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    install_requires=install_requires

)
