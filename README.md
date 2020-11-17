# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a celebrated global optimization technique, which applies steepest descent iterations to an initial starting point in order to find a local minimizer. The METOD algorithm can be more efficient than Multistart as some iterations of steepest descent are stopped early if certain conditions are satisfied. This avoids repeated descents to the same minimizer(s). 

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
3) To run tests, pytest is used which will be installed if step 2 has been completed successfully. In the same directory as step 3, run the following in the command line:
```python
pytest
```

