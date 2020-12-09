from setuptools import find_packages, setup 

exec(open("src/metod/version.py", "r").read())

requirements = ["numpy>=1.16.2", "scipy>=1.2.1", "pytest>=4.3.1",
                "tqdm>=4.32.1", "pandas>=0.24.2", "setuptools>=42.0.2",
                "hypothesis>=5.1.5 ", "SALib>=1.3.12"]

setup(
    name="metod",
    install_requires=requirements,
    version=__version__,
    author="Meg Scammell",
    author_email="scammellm@cardiff.ac.uk",
    license="MIT",
    keywords=["global optimization", "random search", "multistart", "python"],
    packages=find_packages("src"),
    package_dir={"":"src"},
    description="Applying METOD algorithm for a number of starting points.",
)

