------------------------------
Electric Vehicle Charging Data
------------------------------

Raw data comes in column seperated variable format (.csv). Naming convention is: 
(number of ports)_(number of events per port)_(port power in kilowatts).csv

To create a chargin competition with "semi-synthetic" data, we use the 8 event EV loads as these allow 
for more variability in our model due to performativity. The data is split evenly between number of ports and port power
in kilowatts, but not simultaneosly. To create the game, we randomly split the EV load data into two groups of three charging cites. 

Each .csv is a single column of time series data representing the total station power (in watts) with a single entry
for each minute of the year. 

----------
Parameters
----------

We will use a location scale model such that provider one and two have decision-dependent demand models

a = a0 + Ax + By, 
b = b0 + Cx + Dy. 

Since the raw data is a time series, we pick a time (typically during peak hours). Then we sample from the demand at that point 
in time during the day, drawing uniformly from every day in the year. 

Each entry will correspond to a charging cite where the local utilities will be zero for simplicity. The quality
utility parameter gamma is the port power in kilowatts. Hence data will need to be normalized to kilowatts. 

The Elasticity matrices will be determined by 



