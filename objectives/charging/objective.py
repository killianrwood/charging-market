import numpy as np 
from objectives.charging.sampling import sphere_sample
# objective in expectation

def _expected_objective_fn(x,y,A,B,C,D,f,g,r):

    Ea = A*x +B*y
    Eb = C*x + D*y 

    return np.sum(
    f * (x **2) - g * (y ** 2) 
    - (Ea + r)*x + (Eb + r)*y
    )

def _expected_gradient_fn(x,y,A,B,C,D,f,g,r):

    x_gradient = 2*(f - A)*x + (C-B)*y - r
    y_gradient = -2*(g - D)*y + (C-B)*x + r
    
    return x_gradient, y_gradient 

# stochastic objective

def _stochastic_objective_fn(x,y,a,b,f,g,r):

    return np.sum(
    f*(x **2) - g*(y **2) 
    - (a+r)*x + (b+r)*y
    )

def _stochastic_gradient_fn(x,y,a,b,f,g,r):

    x_gradient = 2*f*x - (a + r)
    y_gradient = -2*g*y + (b + r)
    
    return x_gradient, y_gradient

def _equilibrium_gradient_fn(x,y,A,B,C,D,f,g,r):

    x_gradient = 2*f*x - ( A*x + B*y + r)
    y_gradient = -2*g*y + ( C*x + D*y + r)

    return x_gradient, y_gradient

def _zero_order_gradient(x,y,x_dir,y_dir,h,a,b,f,g,r,):

    x_dim, y_dim = x.shape[0], y.shape[0]
    #x_direction, y_direction = sphere_sample(x_dim), sphere_sample(y_dim)

    eval = _stochastic_objective_fn(
        x=x + h * x_dir,
        y=y + h * y_dir,
        a=a,
        b=b,
        f=f,
        g=g,
        r=r)
    
    x_gradient = (x_dim / h) * eval * x_dir
    y_gradient = (y_dim / h) * eval * y_dir

    return x_gradient, y_gradient
