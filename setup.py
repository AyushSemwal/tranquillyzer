from setuptools import setup

setup(
    name='tranquillyzer',
    version='0.1.0',
    packages=['.', 'scripts', 'utils'],
    py_modules=['main'],
    include_package_data=True,
    package_data={
        '': ['models/*.h5', 'utils/*.tsv'],
    },
    install_requires=[
        'numpy', 'pandas', 'polars', 'matplotlib', 'seaborn', 'tqdm',
        'filelock', 'tensorflow', 'tensorflow-addons', 'rapidfuzz',
        'pysam', 'numba', 'typer', 'biopython', 'python-Levenshtein', 'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'tranquillyzer=main:app',
        ],
    },
    python_requires='>=3.10,<3.11',
)