from setuptools import setup, find_packages

def parse_requirements(filename):
    """ Load requirements from a pip requirements file """
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name='hforge',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'my-command=scripts.script1:main',  # if you want to create CLI commands
        ],
    },
    test_suite='tests',
)