import numpy as np
from typing import Callable, Tuple

from objectives.charging.objective import (
_expected_objective_fn,
_stochastic_objective_fn,
_expected_gradient_fn,
_stochastic_gradient_fn,
_equilibrium_gradient_fn,
_zero_order_gradient
)

from objectives.charging.sampling import (_stationary_sample,_demand_sample, sphere_sample)

class ChargingMarket:

    """
    A Charging Market Class.

    Initialize objective with: Elasticity matrices A,B,C,D; expected demand Ea and Eb; charging utility f and g;
    and local utility.

    Instance methods will wrap the parameters and return a callable function for use in optimization.

    Args:
        A,B,C,D: 
            These are elasticity "matrices" used in the location scale families a = a_0 + Ax + By, 
            b = b_0 + Cx + dy. Either use size (n,n) or (n,). At present, we assume they are "diagonal", 
            or just a single value for each region rather than a matrix of values that depend on neighboring areas. 
        Ea,Eb: 
            Expected demand from the distribution.
        f,g: 
            Charging capacity of charging stations at each zone. 
        r: 
            Local utility at each zone.
        P: 
            Distribution matrix of charging demand. Matrix is of size (365, n_stations)


    """

    def __init__(self,A,B,C,D,f,g,r,P,**kwargs):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.f = f.flatten()
        self.g = g.flatten()
        self.r = r.flatten()
        self.P = P
        
        self.x_dim = (self.f).shape[0]
        self.y_dim = (self.g).shape[0]

    def expected_objective(self) -> Callable:

        def caller_fn(x,y):
            return _expected_objective_fn(
                x,
                y,
                self.A,
                self.B,
                self.C,
                self.D,
                self.f,
                self.g,
                self.r
            )
        
        return caller_fn

    def expected_gradient(self) -> Callable:

        def caller_fn(x,y):
            return _expected_gradient_fn(
                x,
                y,
                self.A,
                self.B,
                self.C,
                self.D,
                self.f,
                self.g,
                self.r
            )
            
        return caller_fn

    def equilibrium_gradient(self) -> Callable:

        def caller_fn(x,y):
            return _equilibrium_gradient_fn(
                x,
                y,
                self.A,
                self.B,
                self.C,
                self.D,
                self.f,
                self.g,
                self.r
            )
            
        return caller_fn

    def stochastic_objective(self, n_samples) -> Callable:

        def caller_fn(x,y):
            
            a, b = self.demand_sample(x, y, n_samples)
            
            return _stochastic_objective_fn(
                x,
                y,
                a,
                b,
                self.f,
                self.g,
                self.r
            )

        return caller_fn

    def stochastic_gradient(self,n_samples) -> Callable:

        def caller_fn(x,y):
            
            a, b = self.demand_sample(x, y, n_samples)
            
            return _stochastic_gradient_fn(
                x,
                y,
                a,
                b,
                self.f,
                self.g,
                self.r
            )
        return caller_fn
    
    def zero_order_gradient(self, h) -> Callable:
        
        def caller_fn(x,y):
            
            x_dir, y_dir = sphere_sample(self.x_dim), sphere_sample(self.y_dim)
            a, b = self.demand_sample(x + h*x_dir, y + h* y_dir)
            
            return _zero_order_gradient(
                x=x,
                y=y,
                x_dir=x_dir,
                y_dir=y_dir,
                h=h,
                a=a,
                b=b,
                f=self.f,
                g=self.g,
                r=self.r
            )

        return caller_fn
    

    def demand_sample(self,x,y,n_samples=1) -> Tuple[np.ndarray,np.ndarray]:

        a, b = self.stationary_sample(n_samples)

        return _demand_sample(
        x,y,a,b,self.A,self.B,self.C,self.D
        )


    def stationary_sample(self, n_samples=1) -> Tuple[np.ndarray,np.ndarray]:

        return _stationary_sample(self.P, n_samples)




