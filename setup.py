from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mimo_keras',
    packages=['mimo_keras'],
    version='0.1.4',
    license='MIT',
    description='Data generator for keras multiple-input multiple-output model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ehsan Karimi',
    author_email='ehsankb91@gmail.com',
    url='https://github.com/ehkarimi/mimo_keras',
    download_url='https://github.com/ehkarimi/mimo_keras/archive/V0.1.1.tar.gz',
    keywords=['keras data generator', 'data generator', 'multi-input multi-output model', 'medical image processing'],
    install_requires=[
        'numpy>=1.19.4',
        'pandas>=1.1.0',
        'keras>=2.5.0'
    ],
)
