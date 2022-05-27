from setuptools import setup, find_packages

setup(
    name='fad21',
    version='2022.04.13',
    description='FAD 2021 Scorer Package',
    author='Lukas Diduch',
    author_email='lukas.diduch@nist.gov',
    url='https://openfad.nist.gov',
    packages=find_packages(include=['fad21', 'fad21.*']),
    install_requires=[
        'PyYAML',
        'pandas',
        'numpy',        
        'sklearn',
        'h5py',        
        'matplotlib',        
        'tables',
        'dill'
    ],
    # use python -m pip install -e .[interactive]
    extras_require={'interactive': ['jupyter'],
                    'documentation': ['pydoc-markdown[novella]', 'mkdocs', 'mkdocs-windmill'],
                    'tests': ['pytest']},
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['fad-scorer=fad21.__main__:main']
    }
    #,package_data={'fad21': ['data/input_test.csv']}
)
