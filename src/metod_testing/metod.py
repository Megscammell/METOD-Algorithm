
import warnings
from warnings import warn

import numpy as np
import tqdm

import metod_testing as mtv3


def format_Warning(message, category, filename, lineno, line=''):
    """Converts warning message to a suitable format """
    return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' +str(message) + '\n'

warnings.formatwarning = format_Warning

def metod(f, g, func_args, d, num_points = 1000, beta = 0.01, 
          tolerance = 0.00001, projection = False, const = 0.1, 
          m = 3, option = 'minimize', met='Nelder-Mead', initial_guess = 0.05,
          set_x = np.random.uniform, bounds_set_x = (0,1)):

    """Returns total number of minima discovered by algorithm, position of local minima, function values of local minima and number of unnecessary descents.

    Keyword arguments:
    f -- function
    g -- gradient
    func_args -- arguments passed to f and g
    d -- is the dimension
    num_points -- number of uniform random points generated
    beta -- small constant step size to compute partner points
    tolerance -- stopping condition for steepest descent iterations
    projection -- is a boolean variable. If projection = True, this projects points back to the [0,1]^d cube.
    const --  a constant for the minimum euclidean distance to be larger than to classify a point as a new local minima
    m -- warm up period
    option - choose from 'minimize' or 'minimize_scalar'. Default is 'minimize'.
    met -- Choose appropiate method for option. Default is 'Nelder-Mead'.
    initial_guess -- is passed to the scipy.optimize.minimize function. This is recommended to be small (0.05). Method chosen is Nelder-Mead.
    set_x -- Generates points for the METOD algorithm. Can be either inbuilt numpy.random distribution that generates points, list or numpy array of size num_points
    bounds_set_x - Bounds for numpy.random distribution
    """
    if isinstance(d, int) == False:
        raise ValueError('d must be an integer.')
        
    if isinstance(num_points, int) == False:
        raise ValueError('num_points must be an integer.')
        
    if isinstance(beta, float) == False:
        raise ValueError('beta must be a float.')
        
    if isinstance(tolerance, float) == False:
        raise ValueError('tolerance must be a float.')

    if isinstance(projection, bool) == False:
        raise ValueError('projection must be boolean.')
        
    if isinstance(const, float) == False:
        raise ValueError('const must be a float.')
    
    if isinstance(m, int) == False:
        raise ValueError('m must be an integer.') 

    if isinstance(option, str) == False:
        raise ValueError('option must be a string.') 

    if isinstance(met, str) == False:
        raise ValueError('met must be a string.') 
        
    if isinstance(initial_guess, float) == False:
        raise ValueError('initial_guess must be a float.')  
    
    if d < 2:
        raise ValueError('must have d > 1') 
       
    if m < 1:
        raise ValueError('must have m >= 1')

    if type(set_x) is list or type(set_x) is np.ndarray:
        num_points = len(set_x)
        projection = False
        bound_1 = None
        bound_2 = None
    elif set_x == np.random.uniform:
        bound_1 = bounds_set_x[0]
        bound_2 = bounds_set_x[1]


        
    if beta >= 1:
        warn('beta too high and would require that the largest eigenvalue is smaller than one. Default beta is used.', RuntimeWarning)
        beta = 0.01
        
    if tolerance > 0.1:
        warn('Tolerance is too high and replaced with default.', RuntimeWarning)
        tolerance = 0.00001

    # if num_points < 50:
    #     warn('num_points is very small, so will be replaced by the default.', RuntimeWarning)
    #     num_points = 1000
  
        
    des_x_points = []
    des_z_points = []
    discovered_minimas = []

    if type(set_x) is list or type(set_x) is np.ndarray:
        x = set_x[0]
        if x.shape[0] != d:
            raise ValueError('set_x does not contain points which are of size d')
    else:
        x = set_x(*bounds_set_x,(d,))

    iterations_of_sd, its = mtv3.apply_sd_until_stopping_criteria(
                            x, d, projection, tolerance, option, met, initial_guess, func_args, f, g, bound_1, bound_2)
    if its <= m:
        raise ValueError('m is equal to or larger than the total number of steepest descent iterations to find a minimizer. Please change m or change tolerance.') 
    des_x_points.append(iterations_of_sd)
    discovered_minimas.append(iterations_of_sd[its].reshape(d,))

    sd_iterations_partner_points = mtv3.partner_point_each_sd( iterations_of_sd, d, beta, its, g, func_args)

    des_z_points.append(sd_iterations_partner_points)
    number_minimas = 1
    for remaining_points in tqdm.tqdm(range(num_points - 1)):
        if type(set_x) is list or type(set_x) is np.ndarray:
            x = set_x[remaining_points + 1, :]
            if x.shape[0] != d:
                raise ValueError('set_x does not contain points which are of size d')
        else:
            x = set_x(*bounds_set_x,(d,))

        warm_up_sd, warm_up_sd_partner_points = mtv3.apply_sd_until_warm_up (x, d, m, beta, projection, option, met, initial_guess,func_args, f, g, bound_1, bound_2)
        
        x_1 = warm_up_sd[m - 1].reshape(d,)
        z_1 = warm_up_sd_partner_points[m - 1].reshape(d,)
        x_2 = warm_up_sd[m].reshape(d,)
        z_2 = warm_up_sd_partner_points[m].reshape(d,) 
   
        possible_regions = mtv3.check_alg_cond(number_minimas, x_1, z_1, x_2, z_2, des_x_points, des_z_points, m - 1, d)

        if possible_regions == []:
            iterations_of_sd_part, its = mtv3.apply_sd_until_stopping_criteria(x_2, d, projection, tolerance, option, met, initial_guess, func_args, f, g, bound_1, bound_2)
            if (its + m) <= m:
                raise ValueError('m is equal to or larger than the total number of steepest descent iterations to find a minimizer. Please change m or change tolerance.') 
            iterations_of_sd = np.vstack([warm_up_sd, iterations_of_sd_part[1:,]])
            des_x_points.append(iterations_of_sd)
                
            discovered_minimas.append(iterations_of_sd[its+m].reshape(d,))

            sd_iterations_partner_points = mtv3.partner_point_each_sd(iterations_of_sd, d, beta, its + m, g, func_args)

            des_z_points.append(sd_iterations_partner_points)
            number_minimas += 1

    unique_minima, unique_number_of_minima = mtv3.check_unique_minimas(discovered_minimas, const)
    func_vals_of_minimas = [f(element, *func_args) for element in unique_minima]
                    
    
    return unique_minima, unique_number_of_minima, func_vals_of_minimas, (len(des_x_points) - unique_number_of_minima)

