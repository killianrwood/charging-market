import numpy as np

from typing import Callable


def step_size_scheduler(
    mode: str, 
    multiplier: float = 1.0,
    decay_parameter: float = 0.5,
    decay_schedule: int = 1,
    **kwargs
) -> Callable:

    """
    
    A step size schedule builder.

    Args:
        mode: 
            The step size function mechanism. Currently only "harmonic" and "geometric" are allowed.
        multiplier:
            A constant onstant number used as initial step-size value.
        decay_factor:  
            A step-size function parameter. 
        decay_schedule:
            Number of iterations before decreasing the step_size input by decay_factor.

    """


    mode = mode.lower()

    if mode == "geometric":

        def caller_fn(t):
            
            return geometric_stepsize_schedule(
                t=t,
                multiplier=multiplier,
                decay_scheduele=decay_schedule,
                geometric_factor=decay_parameter
            )
    
    elif mode == "polynomial":

        def caller_fn(t):

            return polynomial_stepsize_schedule(
                t=t,
                multiplier=multiplier,
                decay_factor=decay_parameter,
                decay_schedule=decay_schedule
              )

    elif mode == "polynomial_sqrt":

        def caller_fn(t):

            return polynomial_sqrt_stepsize_schedule(
                t=t,
                multiplier=multiplier,
                decay_factor=decay_parameter,
                decay_schedule=decay_schedule
              )

    else:
        raise NotImplementedError(
            f"Schedueler mode {mode} is not valid."
        )

    return caller_fn


def polynomial_stepsize_schedule(t, multiplier, decay_factor, decay_schedule):
    
    return multiplier / ( decay_factor +  t // decay_schedule)


def polynomial_sqrt_stepsize_schedule(t, multiplier, decay_factor, decay_schedule):
    
    return multiplier / (  decay_factor + np.sqrt(t // decay_schedule) )


def geometric_stepsize_schedule(t, multiplier, geometric_factor,  decay_schedule):

    return multiplier * geometric_factor ** ( t // decay_schedule)

