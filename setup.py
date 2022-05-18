from setuptools import setup
import unittest

setup(
    name='entity_corrector',
    version='0.0.1',
    packages=['entity_corrector'],
    install_requires=[
        "scikit-learn",
        "numpy",
        "pyxdameraulevenshtein",
        "pytest"
    ],
    test_suite="tests"
)

#def my_test_suite():
#    test_loader = unittest.TestLoader()
#    test_suite = test_loader.discover('tests', pattern='test_*.py')
#    return test_suite
