# METOD (Multistart With Early Termination of Descents)-Algorithm-
Multistart is a celebrated global optimization technique which applies steepest descent iterations to a point in order to find a local minimizer. The METOD algorithm can be more efficient that Multistart due to some iterations of steepest descent being stopped early if certain conditions are satisfied. This avoids repeated descents to the same minimizer(s) already discovered. For more information about the METOD algorithm, see https://link.springer.com/article/10.1007/s10898-019-00814-w. 

## Dependencies
To run some of the examples provided, it is required to have Anaconda installed, as some examples are in a Jupyter notebook which acts as a procedure guide. 

The algorithm uses scipy.optimize to calculate the step size, in particular, the user has the option of using  scipy.optimize.minimize, scipy.optimize.minimize_scalar and scipy.optimize.line_search. Documentation on these methods can be found in https://docs.scipy.org/doc/scipy/reference/optimize.html. Analysis of using each of these methods to calculate the step size can be found in ???. The default method used in metod.py is scipy.optimize.minimize with the 'Nelder-Mead' option.

## Installation
To use the METOD Algorithm, please do the following:

- Select 'clone or download' button on the far right of the repository
- Select 'download zip' and save the file
- In some cases, the user may need to unzip the file before use

If you would like to further develop any code (i.e adding different method for step size) or run the tests please do the following.

- Open the terminal and navigate to where you would like the file to be stored
- Run the following code:
```python
git clone https://github.com/Megscammell/METOD-Algorithm.git
```

## Usage
### Jupyter notebooks
The main advantage of using the METOD Algorithm is that it can be more efficient than standard Multistart. 
Once the METOD-Algorithm repository has been installed, the user has access to two examples:

- **METOD Algorithm - Minimum of several quadratic forms**
- **METOD Algorithm - Sum of Gaussians**

Each example is a Jupyter notebook and contains a procedure guide built in detailing what is required from the user in each step. 

**METOD Algorithm - Custom** is a Jupyter notebook which allows the user to define the function (f), gradient (g), function arguments (args) and dimension (d). If the user chooses to do this, it is worth reviewing all input paramaters found in ??? of metod.py to see if any need updating. Possible parameters to update may be beta, tolerance and m.

### Other
The METOD algorithm is contained in metod.py, so as long as the required parameters are defined, it is possible to run metod.py without using a Jupyter notebook. An example of this is main.py.


## Outputs
metod.py outputs the positions of the discovered minima, total number of discovered minima, function values of each discovered minima and also the number of extra descents. More information on these outputs can be found in ???.

All outputs for each of the three Jupyter notebooks will be saved to csv files in the same folder to where the notebook is run. For **METOD Algorithm - Minimum of several quadratic forms** and **METOD Algorithm - Sum of Gaussians**, there is a test built in at the end (which is optional to run) that checks each of the discovered minima.

## Possible Enhancements
- sd_iteration.py is the python module that calculates the step size for a given point. If the user wishes to use a different method to calculate step size, they may update sd_iteration.py to include the required method. It is recommended to carry out testing of the new method via pytest before implementing. 
