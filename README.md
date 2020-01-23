# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a celebrated global optimization technique which applies steepest descent iterations to a point in order to find a local minimizer. The METOD algorithm can be more efficient that Multistart due to some iterations of steepest descent being stopped early if certain conditions are satisfied. This avoids repeated descents to the same minimizer(s) already discovered. 

## Installation
To use the METOD Algorithm, please do the following:
1) Ensure git is installed. Instructions on how to do this can be found on \\https://git-scm.com/book/en/v2/Getting-Started-Installing-Git.

2) Open the command line and navigate to where you would like the file to be stored and run the following code:
```python
git clone https://github.com/Megscammell/METOD-Algorithm.git
```
3) Navigate to the directory that contains the setup.py file and run the following code:
```python
python setup.py develop
```
4) To run tests, pytest is used which will be installed if step 3 has been completed successfully. In the same directory as step 3, run the following in the command line:
```python
pytest
```

## Dependencies
All required libraries for metod.py are listed in requirements.txt. If installation is completed successfully, then all required libraries will be automatically installed. Examples using the METOD algorithm are available as Jupyter notebooks and as a Python file. To run some of the examples provided in Jupyter notebooks, it is required to have Anaconda installed. 

## Usage
Two examples are Jupyter notebooks called **METOD Algorithm - Minimum of several quadratic forms** and **METOD Algorithm - Sum of Gaussians**. Within each notebook is a procedure guide detailing what parameters need to be updated and in what order the code should be run. Outputs from metod.py will be saved to csv files and stored within the same directory as the notebook.

Alternatively, **main.py** found in src/metod_testing can be used to run metod.py. The **main.py** program can be updated in the following way:


1) Update the function (f), gradient (g) and dimension (d). Defaults are the minimum of several quadratic forms function and gradient with dimension 20.

2) Due to the minimum of several quadratic forms function and gradient being chosen, the following parameters are required; number of local minima (P), smallest eigenvalue (lambda\_1) and largest eigenvalue (lambda\_2). These are used to create random function arguments from function\_parameters\_quad.py used solely by the minimum of several quadratic forms function and gradient. If a different function and gradient is used, this part of the code will need to be changed in order to produce function arguments for the chosen function and gradient. 

3) The METOD algorithm solver will be run for the choice of function, gradient, dimension and function arguments. If any of the optional input parameters need to be changed for metod.py, they will need to be changed at this point of the main.py program.

4) Outputs from metod.py will be saved as csv files in the same directory as main.py.
