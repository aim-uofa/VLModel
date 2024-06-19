from setuptools import setup, find_packages

# Read contents of the requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='hawkllama',
    version='0.1.0',
    author='Hengtao Li',
    author_email='liht@zju.edu.cn',
    description='hawkllama package',
    packages=find_packages(),
    install_requires=requirements,  # Include dependencies from requirements.txt
)