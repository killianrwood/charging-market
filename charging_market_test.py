import os
import numpy as np
import pandas as pd
import pickle as pkl

from numpy.linalg import norm
import matplotlib.pyplot as plt
from os.path import join

from objectives.charging.sampling import build_sampler
from objectives.charging.charging_market import ChargingMarket
from objectives.charging.constraints import interval_projection, primal_dual_projection
from optimization.primal_dual import primal_dual
from objectives.charging.sampling import box_sample
from optimization.step_size_scheduler import step_size_scheduler

"""
Get data from processed_data folder

If data has not been processed yet, please run process_data.py first.

"""

directory = os.path.join(
    "data",
    "processed_data",
    "eight_event"
)
time = 12
data, f, g = build_sampler(
    directory,
    time,
    process=True, 
    return_mean=False
)

days, total_stations = data.shape
n_stations = total_stations // 2
r = np.zeros((n_stations,))  + 0.1  
average_speed = np.mean(np.stack((f,g),axis=1), axis=1)
r = 0.001 * np.array([ 1.0 if mean >=2 else 0.1 for mean in average_speed])
# To create a charging competition, we need A, B, C, D, Ea, Eb, f, g, and r

f = np.ones((n_stations))
g = f.copy()

# To create a charging competition, we need A, B, C, D, Ea, Eb, f, g, and r
A = -0.3*np.ones((n_stations,)) # elasticity to price x for demand of x
C = 0.3*np.ones((n_stations,))                      # elasticity to price x for demand of y
B = C                   # elasticity to price y for demand of x
D = A                       # elastcity to price y for demand of 


gamma = np.min(np.minimum(f,g))
L = np.max(np.maximum(f,g))
epsilon = np.max(np.abs(A))

gap = gamma - ( epsilon * L )

step_size_bound = min(
    gap/(2*(1+epsilon) ** 2 *L ** 2),
    1/gap
)
condition_number = gamma - epsilon*L
ell_bound = 2 * condition_number
kappa_bound = ( (1+epsilon) ** 2 * L ** 2 ) / condition_number

charging = ChargingMarket(
    A=A,
    B=B,
    C=C,
    D=D,
    f=f,
    g=g,
    r=r,
    P=data
)

lower_bound, upper_bound = -1, 2 * 1
projection_fn = lambda x,y: primal_dual_projection(
    x,
    y,
    a=lower_bound,
    b=upper_bound,
    c=lower_bound,
    d=upper_bound
)


"""
Setup algorithms

"""

# equilibrium points

step_size = 1e-3
max_iterations = 2 * 10 **3
# x0 = box_sample(lower_bound, upper_bound,n_stations)
# y0 = box_sample(lower_bound, upper_bound,n_stations)
x0 = 0.5 * np.array([1,-1,1])
y0 = - x0.copy()
# x0 = np.zeros((n_stations,))
# y0 = np.zeros((n_stations,))

equilibrium_gradient = charging.equilibrium_gradient()

x, y, x_values, y_values = primal_dual(
    gradient_fn=equilibrium_gradient,
    projection_fn=projection_fn,
    initial_args=(x0,y0),
    step_size=5e-3,
    max_iterations=10 ** 4, 
    history=True
)

equilibrium_point = np.concatenate((x,y))
z_values = np.concatenate((x_values, y_values),axis=1)
eq_error = norm(equilibrium_point-z_values, axis=1)

# stochastic primal-dual algorithm 


n_samples = 1
stochastic_gradient = charging.stochastic_gradient(n_samples)
scheduler = step_size_scheduler(
    mode = "polynomial",
    multiplier = ell_bound + 0.1,
    decay_parameter = kappa_bound + 0.1,
    decay_schedule= 1
    )
x, y, x_values, y_values = primal_dual(
    gradient_fn=stochastic_gradient,
    projection_fn=projection_fn,
    initial_args=(x0,y0),
    max_iterations= 10 ** 4, 
    history=True, 
    step_size_fn=scheduler
)

z_values = np.concatenate((x_values, y_values),axis=1)
stoch_error = norm(equilibrium_point-z_values, axis=1) ** 2


# saddle point algorithms 

saddle_gradient = charging.expected_gradient()
x, y, x_values, y_values = primal_dual(
    gradient_fn=saddle_gradient,
    projection_fn=projection_fn,
    initial_args=(x0,y0),
    step_size=step_size,
    max_iterations= 10 ** 4, 
    history=True
)

saddle_point = np.concatenate((x,y))
z_values = np.concatenate((x_values, y_values),axis=1)
saddle_error = norm(saddle_point-z_values, axis=1)

from optimization.step_size_scheduler import step_size_scheduler

h = 0.05
dilated_projection_fn = lambda x,y: primal_dual_projection(
    x,
    y,
    a=(1-h)*lower_bound,
    b=(1-h)*upper_bound,
    c=(1-h)*lower_bound,
    d=(1-h)*upper_bound
)
zero_order_gradient = charging.zero_order_gradient(h=h)
scheduler = step_size_scheduler(
    mode = "polynomial",
    multiplier = 1/condition_number,
    decay_parameter = 2.0
    )

x, y, x_values, y_values = primal_dual(
    gradient_fn=zero_order_gradient,
    projection_fn=dilated_projection_fn,
    initial_args=(x0,y0),
    max_iterations= 10 ** 4, 
    history=True,
    step_size_fn = scheduler
)
delta_point = np.concatenate((x,y))
z_values = np.concatenate((x_values, y_values), axis=1)
delta_error = norm(delta_point-z_values, axis=1) **2
saddle_stoch_error = norm(saddle_point-z_values, axis=1)

"""
Print Results

"""

print("RESULTS:")
print(f"Found saddle point at: \n---------------------------")
print(saddle_point)
print(f"Found equilibrium point at: \n---------------------------")
print(equilibrium_point)
print(f"Solution gap is {norm(saddle_point-equilibrium_point)}")



""" 
Make Player 1 figure from processed Data

"""

markers = [ ":","--", "-", ]
alpha_vals = [0.95, 0.5, 0.75]
fig, ax = plt.subplots(figsize=(6,4))
plot_data = data[:,:3]
for station, demand in enumerate(plot_data.T):
    plt.plot(
        demand,
        markers[station],
        color="k",
        linewidth=2,
        label=f"station {station+1}",
        alpha=alpha_vals[station]
        )




plt.rc('legend',fontsize=10)
plt.rc(
    'axes',
     titlesize=10,
     labelsize=10
     )
plt.rc('ytick', labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('figure', titlesize=12)

ax.legend(loc="lower center", ncol=3)
plt.xlabel("day")
plt.ylabel("demand")
plt.title("Demand change between 12-1 pm")
plt.grid("True")
plt.xlim(0,365)
plt.ylim(-1.0,1.0)
plt.show()
fig.savefig(
    join("figures", "demand_versus_day_highres.png"),
    dpi=600
    )



"""

Make Error Plot

"""
# curves = ( stoch_error, delta_error, saddle_stoch_error)
fig, ax = plt.subplots(figsize=(6,4))

ax.plot(
    eq_error,
    linestyle="-",
    linewidth=2,
    label="EPD",
    color = "k",
    alpha=0.95
    )

ax.plot(
    stoch_error,
    linestyle="-.",
    linewidth=2,
    label="SEPD",
    color = "k",
    alpha=0.75
    )

ax.plot(
    delta_error,
    linestyle=":",
    linewidth=2,
    label="$DFO \ to \ z^{*}$",
    color = "k", 
    alpha = 0.65
    )

ax.plot(
    saddle_stoch_error,
    linestyle="--",
    linewidth=2,
    label="$DFO \ to \ z^{*}_{\delta}$",
    color="k",
    alpha=0.45
    )

plt.rc('legend',fontsize=10)
plt.rc(
    'axes',
     titlesize=12,
     labelsize=12
     )
plt.rc('ytick', labelsize=11)
plt.rc('xtick', labelsize=11)
plt.rc('figure', titlesize=12)

ax.legend(loc="lower center", ncol=4)
plt.yscale("log")
plt.xlabel("iterations")
plt.ylabel("error")
plt.title("Algorithm Error per iteration")
plt.xlim(0, 3 * 10 ** 3)
plt.ylim(10**-5, 10 **1 )


plt.grid("False")
plt.show()
fig.savefig(
    join("figures", "error_plot.png"),
    dpi=600
    )
