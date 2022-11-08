import numpy as np
from typing import Callable, Tuple

def primal_dual(
    gradient_fn: Callable, 
    initial_args: Tuple,
    projection_fn: Callable = None,
    max_iterations: int = 10 **3,
    step_size: float = 1e-1, 
    step_size_fn: Callable = None,
    history: bool = True,
    **kwargs
):

    """
    A Primal-Dual Optimization Algorithm. 

    Args:
        gradient_fn: 
            A callable gradient function handle. Should be a function of both the primal and dual variables
            "x" and "y" seperately.
        initial_args: 
            A tuple of primal and dual variable inital conditions (x0,y0).
        projection_fn:
            An optional callable projection function handle. Should be a function of both the primal and dual variables
            "x" and "y" seperately. Defaulted to None value.
        max_iterations:
            Maximum number of iterations before exiting the algorithm. For now, we only use this as our stopping
            condition.
        step_size:
            A starting step-size value.
        step_size_fn"
            An optional callable function that returns a value "step_size" when given a single integer input. If 
            provided, this ignores the step_size argument provided in the function call. 
        history:
            A True/False value indicating whether or not the function should store the numpy updates to the primal
            and dual variables after each iteration.
    
    """
    
    max_iterations = int(max_iterations)

    if projection_fn is not None:
        x, y = projection_fn(*initial_args)
    else: 
        x, y = initial_args
    
    if history:
        x_dim = x.shape[0]
        y_dim = y.shape[0]
        x_values = np.zeros((max_iterations+1,x_dim))
        y_values = np.zeros((max_iterations+1,y_dim))
        x_values[0], y_values[0] = x, y
    
    for t in range(max_iterations):

        if step_size_fn is not None:
            step_size = step_size_fn(t)
        
        x_grad, y_grad = gradient_fn(x,y)
        x -=step_size*x_grad
        y +=step_size*y_grad

        if projection_fn is not None:
            x, y  = projection_fn(x,y)
        
        if history:
            x_values[t+1], y_values[t+1] = x, y
            
    return x, y, x_values, y_values
