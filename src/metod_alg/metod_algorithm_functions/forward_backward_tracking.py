import numpy as np


def forward_tracking(point, t, f_old, f_new, grad, const_forward,
                     forward_tol, f, func_args):
    """
    Repeatedly multiply step size t by const_forward (where
    const_forward > 1) until either forward_tol is met or until
    the function value cannot be improved any further.

    Parameters
    ----------
    point : 1-D array of shape (d,).
    t : float
        Initial guess of step size.
    f_old : float
            Function value at point. That is,

            `f(point, *func_args) -> f_old`

    f_new : float
            That is,

            `f(point - t *  grad, *func_args) -> f_new`

           where func_args is a tuple of arguments needed to compute the
           function value.
    grad : 1-D array of shape (d,)
            Gradient at point. That is,

            `g(point, *func_args) -> grad`

           where func_args is a tuple of arguments needed to compute the
           gradient.
    const_forward : float
                    The initial guess of the
                    step size will be multiplied by const_forward at each
                    iteration of forward tracking. That is,
                    t * const_back -> t
                    It should be noted that const_forward > 1.
    forward_tol : float
                  It must be ensured that the step size computed by forward
                  tracking is not greater than forward_tol. If this is the
                  case, iterations of forward tracking are terminated.
                  Typically, forward_tol is a very large number.
    f : objective function.

        `f(point, *func_args) -> float`

        where func_args is a tuple of arguments needed to compute the
        function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    track : 2-D array
            The first column of track contains the step size at each iteration
            of forward tracking. The second column of track contains the
            function value at each iteration of forward tracking. For example,
            [[0, f(point - 0 * grad, *func_args)],
             [t, f(point - t * grad, *func_args)],
             [t * const_forward,
              f(point - t * const_forward * grad, *func_args)],
             ...
             [t * (const_forward) ^ i,
              f(point - t * (const_forward) ^ i * grad, *func_args)]],
            where i is the number of iterations of forward tracking.
    flag : boolean
           True if  the function value cannot be improved any further.
           False if forward_tol is met.
    """
    assert(const_forward > 1)
    track = np.array([[0, f_old], [t, f_new]])
    t = t * const_forward
    track = np.vstack((track,
                       np.array([t, f(np.copy(point) -
                                 t * grad, *func_args)])))
    while track[-2][1] > track[-1][1]:
        t = t * const_forward
        if t > forward_tol:
            return track, False
        track = np.vstack((track,
                           np.array([t, f(np.copy(point) -
                                     t * grad, *func_args)])))
    return track, True


def backward_tracking(point, t, f_old, f_new, grad, const_back,
                      back_tol, f, func_args):
    """
    Repeatedly multiply step size t by const_back (where const_back < 1)
    until either back_tol is met or until the function value is smaller than
    f(point, *func_args).

    Parameters
    ----------
    point : 1-D array of shape (d,).
    t : float
        Initial guess of step size.
    f_old : float
            Function value at point. That is,

            `f(point, *func_args) -> f_old`

    f_new : float
            That is,

            `f(point - t *  grad, *func_args) -> f_new`

           where func_args is a tuple of arguments needed to compute the
           function value.
    grad : 1-D array of shape (d,)
            Gradient at point. That is,

            `g(point, *func_args) -> grad`

           where func_args is a tuple of arguments needed to compute the
           gradient.
    const_back : float
                 If backward tracking is required, the initial guess of the
                 step size will be multiplied by const_back at each iteration
                 of backward tracking. That is,
                 t * const_back -> t
                 It should be noted that const_back < 1.
    back_tol : float
               It must be ensured that the step size computed by backward
               tracking is not smaller than back_tol. If this is the case,
               iterations of backward tracking are terminated. Typically,
               back_tol is a very small number.
    f : objective function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    track : 2-D array
            If back_tol is not met, then
            the first column of track contains the step size at each iteration
            of backward tracking. The second column of track contains the
            function value at each iteration of backward tracking. For example,
            [[0, f(point - 0 * grad, *func_args)],
             [t, f(point - t * grad, *func_args)],
             [t * const_back, f(point - t * const_back * grad, *func_args)],
             ...
             [t * (const_back) ^ i,
              f(point - t * (const_back) ^ i * grad, *func_args)]],
            where i is the number of iterations of backward tracking.
            Otherwise, if back_tol is met, then track will be of the following
            form
             [[0, f(point - 0 * grad, *func_args)],
             [t, f(point - t * grad, *func_args)]]
    """
    assert(const_back < 1)
    track = np.array([[0, f_old], [t, f_new]])
    temp_track = np.copy(track)
    while track[0][1] <= track[-1][1]:
        t = t * const_back
        if t < back_tol:
            return temp_track
        else:
            track = np.vstack((track,
                               np.array([t, f(np.copy(point) -
                                         t * grad, *func_args)])))
    return track


def compute_coeffs(track_y, track_t):
    """
    Least squares polynomial fit to obtain step size.

    Parameters
    ----------
    track_y : 1-D array with shape (3, )
              Array containing the three smallest function values from
              applying either forward or backward tracking.
              Must have track_y[0] > track_y[1] and
              track_y[2] > track_y[1].
    track_t : 1-D array with shape (3, )
              Array containing the step sizes corresponding to the three
              smallest function values from applying either forward or
              backward tracking. Must have
              track_t[0] < track_t[1] < track_t[2]

    Returns
    -------
    res : float
          Minimum value of the polynomial fit.
    """
    coeffs = np.polyfit(track_t, track_y, 2)
    res = -coeffs[1]/(2 * coeffs[0])
    assert((res >= 0))
    return res


def arrange_track_y_t(track, track_method):
    """
    Select three step sizes, for example (t1, t2, t3), such that t1 < t2 < t3,
    f(t1) > f(t2) and f(t2) < f(t3). The three selected step sizes and
    corresponding function values will be used in compute_coeffs().

    Parameters
    ----------
    track : 2-D array
            First column of track contains the step size at each iteration
            of backward or forward tracking. The second column of track
            contains the function value at each iteration of backward or
            forward tracking.
    track_method : string
                   Either 'Backward' or 'Forward', depending on whether
                   backward or forward tracking was applied.

    Returns
    --------
    track_y : 1-D array with shape (3, )
              Array containing the three smallest function values from
              applying either forward or backward tracking.
              Must have track_y[0] > track_y[1] and
              track_y[2] > track_y[1].
    track_t : 1-D array with shape (3, )
              Array containing the step sizes corresponding to the three
              smallest function values from applying either forward or
              backward tracking. Must have
              track_t[0] < track_t[1] < track_t[2]
    """
    track_y = track[:, 1]
    track_t = track[:, 0]
    if len(track_y) > 3:
        if track_method == 'Backward':
            min_pos = np.argmin(track_y)
            prev_pos = min_pos - 1
            track_y = np.array([track_y[0],
                                track_y[min_pos],
                                track_y[prev_pos]])
            track_t = np.array([track_t[0],
                                track_t[min_pos],
                                track_t[prev_pos]])
            assert(track_t[0] < track_t[1])
            assert(track_t[2] > track_t[1])
        else:
            min_pos = np.argmin(track_y)
            next_pos = np.argmin(track_y[min_pos:]) + (min_pos + 1)
            prev_pos = np.argmin(track_y[:min_pos])
            track_y = np.array([track_y[prev_pos],
                                track_y[min_pos],
                                track_y[next_pos]])
            track_t = np.array([track_t[prev_pos],
                                track_t[min_pos],
                                track_t[next_pos]])
            assert(track_t[0] < track_t[1])
            assert(track_t[2] > track_t[1])
    return track_y, track_t


def check_func_val_coeffs(track, track_method, point, grad, f, func_args):
    """
    Find the step size opt_t using compute_coeffs() and check that
    f(point - opt_t *grad, *func_args) is less than the smallest
    function value found so far. The step size corresponding the
    the best function value is returned.

    Parameters
    ----------
    track : 2-D array
            First column of track contains the step size at each iteration
            of backward or forward tracking. The second column of track
            contains the function value at each iteration of backward or
            forward tracking.
    track_method : string
                   Either 'Backward' or 'Forward', depending on whether
                   backward or forward tracking was applied to obtain track.
    point : 1-D array of shape (d,).
    grad : 1-D array of shape (d,)
            Gradient at point. That is,

            `g(point, *func_args) -> grad`

           where func_args is a tuple of arguments needed to compute the
           gradient.
    f : objective function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    --------
    opt_t : step size corresponding to best function value found.
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
    Apply either forward or backward tracking to obtain a step size for a
    steepest descent iteration.

    Forward tracking is applied if f(point, *func_args) >
    f(point - t * grad, *func_args). Otherwise, backward tracking is applied.

    Parameters
    ----------
    point : 1-D array of shape (d,).
    f_old : float
            Function value at point. That is,

            `f(point, *func_args) -> f_old`
            where func_args is a tuple of arguments needed to compute the
            function.
    grad : 1-D array of shape (d,)
            Gradient at point. That is,

            `g(point, *func_args) -> grad`

           where func_args is a tuple of arguments needed to compute the
           gradient.
    t : float
        Initial guess of step size.
    const_back : float
                 If backward tracking is required, the initial guess of the
                 step size will be multiplied by const_back at each iteration
                 of backward tracking. That is,
                 t * const_back -> t
                 It should be noted that const_back < 1.
    back_tol : float
               It must be ensured that the step size computed by backward
               tracking is not smaller than back_tol. If this is the case,
               iterations of backward tracking are terminated. Typically,
               back_tol is a very small number.
    const_forward : float
                    If forward tracking is required, the initial guess of the
                    step size will be multiplied by const_forward at each
                    iteration of forward tracking. That is,
                    t * const_back -> t
                    It should be noted that const_forward > 1.
    forward_tol : float
                  It must be ensured that the step size computed by forward
                  tracking is not greater than forward_tol. If this is the
                  case, iterations of forward tracking are terminated.
                  Typically, forward_tol is a very large number.
    f : objective function.

        `f(point, *func_args) -> float`

        where point` is a 1-D array with shape(d, ) and func_args is
        a tuple of arguments needed to compute the function value.
    func_args : tuple
                Arguments passed to the function f.

    Returns
    -------
    opt_t : float
            Step size for a steepest descent iteration.
    """
    f_new = f(np.copy(point) - t * grad, *func_args)
    if f_old <= f_new:
        track_method = 'Backward'
        track = backward_tracking(point, t, f_old, f_new,
                                  grad, const_back, back_tol, f,
                                  func_args)
        if len(track) == 2:
            return 0
        else:
            opt_t = (check_func_val_coeffs
                     (track, track_method,
                      point, grad, f, func_args))
            return opt_t

    elif f_old > f_new:
        track_method = 'Forward'
        track, flag = (forward_tracking
                       (point, t, f_old, f_new, grad,
                        const_forward, forward_tol, f, func_args))
        if flag == False:
            t = track[-1][0]
            return t
        else:
            opt_t = (check_func_val_coeffs
                     (track, track_method,
                      point, grad, f, func_args))
            return opt_t
