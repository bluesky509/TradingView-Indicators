# setup.py
from setuptools import setup, find_packages

setup(
    name='TradingViewIndicators',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas'
        # Add other dependencies if necessary
    ],
    description='A collection of TradingView indicators implemented in Python',,
    url='https://github.com/bluesky509/TradingView-Indicators',
)
