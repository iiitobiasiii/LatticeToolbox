from setuptools import setup, find_packages

setup(
    name='lattice_toolbox',
    python_requires='>3.5.2',
    version='0.1',
    packages=find_packages(include=['latticetoolbox', 'latticetoolbox.*']),
    package_data={'latticetoolbox':['latticetoolbox/resources/latticedicts/*.pkl']},
    url='',
    license='MIT',
    author='tobias',
    author_email='kontakt@tobiasbusse.de',
    description='A Toolbox for Triangular and Hexagonal Lattices',
    install_requires =['matplotlib>=2.2.0',
                       "numpy>=1.14.5",
                       'networkx>=2.5',]
)
