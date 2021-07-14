from setuptools import find_packages, setup 

exec(open("src/metod_alg/version.py", "r").read())

requirements = ["numpy>=1.16.2", "scipy>=1.2.1", "pytest>=4.3.1",
                "tqdm>=4.32.1", "pandas>=0.24.2", "setuptools>=42.0.2",
                "hypothesis>=5.1.5 ", "SALib>=1.3.12"]

# with open("README.MD", "r") as f:
#     long_description = f.read()

setup(
    name="metod_alg",
    install_requires=requirements,
    version=__version__,
    author="Meg Scammell",
    author_email="scammellm@cardiff.ac.uk",
    license="MIT",
    url="https://github.com/Megscammell/METOD-Algorithm.git",
    keywords=["global optimization", "random search", "multistart", "python"],
    packages=find_packages("src"),
    package_dir={"":"src"},
    description="Applying METOD algorithm for a number of starting points.",
    # long_description=long_description,
    # long_description_content_type="text/x-MD",
)

