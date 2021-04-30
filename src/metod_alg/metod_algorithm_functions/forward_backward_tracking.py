import numpy as np
import scipy
from scipy import optimize

from metod_alg import metod_algorithm_functions as mt_alg


def compute_forward(t, const_forward, forward_tol, track, point, grad, 
                    f, func_args):
    """
    Increases step size by a multiple of const_forward (greater
    than one) until either forward_tol is met or until the function value cannot
    be improved any further.
    """
    count_func_evals = 0
    while track[-2][1] > track[-1][1]:
        t = t * const_forward
        if t > forward_tol:
            return track, count_func_evals, False
        track = np.vstack((track, np.array([t, f(np.copy(point) - t * grad, *func_args)])))
        count_func_evals += 1
    return track, count_func_evals, True


def forward_tracking(point, t, f_old, f_new, grad, const_forward,
                     forward_tol, f, func_args):
    """
    Obtains step size from compute_forward() and returns outputs.
    """
    assert(const_forward > 1)
    track = np.array([[0, f_old], [t, f_new]])    
    t = t * const_forward
    track = np.vstack((track, np.array([t, f(np.copy(point) - t * grad, *func_args)])))
    total_func_evals = 1
    track, count_func_evals, flag = (compute_forward
                                     (t, const_forward, forward_tol,   
                                      track, point, grad, f, func_args))
    total_func_evals += count_func_evals
    return track, total_func_evals, flag


def backward_tracking(point, t, f_old, f_new, grad, const_back, 
                      back_tol, f, func_args):
    """
    Decreases step size by a multiple of const_backward (less than one) until
    either back_tol is met or until the function value is smaller than
    f(point, *func_args).
    """
    assert(const_back < 1)
    count_func_evals = 0
    track = np.array([[0, f_old], [t, f_new]])
    temp_track = np.copy(track)
    while track[0][1] <= track[-1][1]:
        t = t * const_back
        if t < back_tol:
            return temp_track, count_func_evals
        else:
            track = np.vstack((track, np.array([t, f(np.copy(point) - t * grad, *func_args)])))
            count_func_evals += 1
    return track, count_func_evals


def compute_coeffs(track_y, track_t):
    """
    Performs least squares polynomial fit to obtain step size.
    """
    coeffs = np.polyfit(track_t, track_y, 2)
    assert((-coeffs[1]/(2 * coeffs[0]) >= 0))
    return -coeffs[1]/(2 * coeffs[0])


def arrange_track_y_t(track, track_method):
    """
    Select three step sizes, for example (t1, t2, t3), such that t1 < t2 < t3,
    f(t1) > f(t2) and f(t2) < f(t3). The three selected step sizes and
    corresponding function will be used in compute_coeffs().
    """    
    track_y = track[:, 1]
    track_t = track[:, 0]
    if len(track_y) > 3:
        if track_method == 'Backward':
            min_pos = np.argmin(track_y)
            prev_pos = min_pos - 1
            track_y = np.array([track_y[0], track_y[min_pos], track_y[prev_pos]])
            track_t = np.array([track_t[0], track_t[min_pos], track_t[prev_pos]])
            assert(track_t[0] < track_t[1])
            assert(track_t[2] > track_t[1])
        else:
            min_pos = np.argmin(track_y)
            next_pos = np.argmin(track_y[min_pos:]) + (min_pos + 1)
            prev_pos = np.argmin(track_y[:min_pos])
            track_y = np.array([track_y[prev_pos], track_y[min_pos], track_y[next_pos]])
            track_t = np.array([track_t[prev_pos], track_t[min_pos], track_t[next_pos]])
            assert(track_t[0] < track_t[1])
            assert(track_t[2] > track_t[1])
    return track_y, track_t


def check_func_val_coeffs(track, track_method, point, grad, f, func_args):
    """
    Compute the step size opt_t using compute_coeffs() and check that
    f(point - opt_t *grad, *func_args) is less than the smallest
    function value found so far. If this is not the case, the
    step size corresponding the the best found function value is
    returned. 
    """    
    track_y, track_t = arrange_track_y_t(track, track_method)
    opt_t = compute_coeffs(track_y, track_t)
    upd_point = np.copy(point) - opt_t * grad
    
    if (f(upd_point, *func_args) > track_y[1]):
        return track_t[1]
    else:  
        return opt_t


def combine_tracking(point, f_old, grad, t, const_back, back_tol,
                     const_forward, forward_tol, f, func_args):
    """
    Compare f_new and f_old to determine whether backward or forward tracking
    is required. For backward tracking, if back_tol is met then a step size of
    0 is returned. Otherwise, check_func_val_coeffs() is called and outputs
    are returned. For forward tracking, if forward_tol is met, the step size 
    corresponding to the best function value is returned. Otherwise, 
    check_func_val_coeffs() is called and outputs are returned.
    """    
    f_new = f(np.copy(point) - t * grad, *func_args)
    total_func_evals = 1
    if f_old <= f_new:
        track_method = 'Backward'
        track, func_evals = backward_tracking(point, t, f_old, f_new, 
                                              grad, const_back, back_tol, f, 
                                              func_args)
        total_func_evals += func_evals
        if len(track) == 2:
            return 0
        else:
            opt_t = (check_func_val_coeffs
                     (track, track_method, 
                      point, grad, f, func_args))
            total_func_evals += 1                                          
            return opt_t

    elif f_old > f_new:
        track_method = 'Forward'
        track, func_evals, flag = (forward_tracking
                                   (point, t, f_old, f_new, grad, 
                                    const_forward, forward_tol, f, func_args))
        total_func_evals += func_evals
        if flag == False:
            t = track[-1][0]
            return t
        else:
            opt_t = (check_func_val_coeffs
                    (track, track_method, 
                     point, grad, f, func_args))
            total_func_evals += 1                                           
            return opt_t

