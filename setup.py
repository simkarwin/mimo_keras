from setuptools import setup
from os import path

VERSION = '0.1.17'
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mimo_keras',
    packages=['mimo_keras'],
    version=VERSION,
    license='MIT',
    description='A DataGenerator for Keras multiple-input multiple-output models and massive datasets with any type of data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Simkarwin',
    author_email='simkarwin@gmail.com',
    url='https://github.com/simkarwin/mimo_keras',
    download_url= f'https://github.com/simkarwin/mimo_keras/archive/refs/tags/v{VERSION}.zip',
    keywords=['keras data generator', 'data generator', 'multi-input multi-output model', 'medical image processing'],
    install_requires=[
        'numpy>=1.0.0',
        'pandas>=1.0.0',
        'keras>=2.0.0'
    ],
)
