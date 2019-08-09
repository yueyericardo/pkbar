from setuptools import setup, find_packages

setup(
    name='pkbar',
    description='Keras Progress Bar for PyTorch',
    url='https://github.com/yueyericardo/pkbar',
    author='Richard Xue',
    license='Apache License 2.0',
    author_email='yueyericardo@gmail.com',
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'numpy',
    ]
)
