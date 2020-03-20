# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a celebrated global optimization technique which applies steepest descent iterations to a point in order to find a local minimizer. The METOD algorithm can be more efficient that Multistart due to some iterations of steepest descent being stopped early if certain conditions are satisfied. This avoids repeated descents to the same minimizer(s) already discovered. 

## Installation
To use the METOD Algorithm, please do the following:

1) Open the command line and navigate to where you would like the file to be stored and run the following code:
```python
git clone https://github.com/Megscammell/METOD-Algorithm.git
```
2) Navigate to the directory that contains the setup.py file and run the following code:
```python
python setup.py develop
```
3) To run tests, pytest is used which will be installed if step 3 has been completed successfully. In the same directory as step 3, run the following in the command line:
```python
pytest
```

## Dependencies
All required libraries for metod.py are listed in requirements.txt. If installation is completed successfully, then all required libraries will be automatically installed. Examples using the METOD algorithm are available as Jupyter notebooks and as a Python file. To run some of the examples provided in Jupyter notebooks, it is required to have Anaconda installed. 

## Usage
Two examples are Jupyter notebooks called **METOD Algorithm - Minimum of several quadratic forms** and **METOD Algorithm - Sum of Gaussians**. Within each notebook is a procedure guide detailing what parameters need to be updated and in what order the code should be run. Outputs from metod.py will be saved to csv files and stored within the same directory as the notebook. Alternatively, main.py found in src/metod_testing can be used to run the METOD algorithm. 


