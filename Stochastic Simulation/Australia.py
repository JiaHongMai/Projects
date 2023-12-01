# -*- coding: utf-8 -*-

'''
#Dataset:
Historical disaster statistics in Australia from 1967 to 2014.

#Project Description:
This project involves the analysis and visualization of historical data on natural disasters in Australia,
specifically focusing on the number of catastrophes per year. A Poisson process is simulated to model the occurrence of disasters over time.
The code is implemented in Python using libraries such as NumPy, Matplotlib, and Pandas.

'''
#Load libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = "auscathist.xlsx"
df = pd.read_excel(data)

#%% Histogram

# Original dataset
x1 = df["Year"]

# Create a larger figure
plt.figure(figsize=(18, 12))

# Plot histogram for the original dataset
hist = plt.hist(x1, bins=len(set(x1)), edgecolor='black', color="#E6A000")
plt.ylabel('Number of Catastrophes', fontsize=10)
plt.title('Number of Natural Disasters per Year in Australia', fontsize=20)

# Set ticks to show every increment on both x and y axes
highest_count = int(max(hist[0]))
plt.xticks(range(min(x1), max(x1)), fontsize=10, rotation=35)
plt.yticks(range(0, highest_count), fontsize=5)

# Linear Regression
counts, bins, _ = hist

from sklearn.linear_model import LinearRegression

X = bins[:-1].reshape(-1, 1)
y = counts.reshape(-1, 1)
model = LinearRegression().fit(X, y)

intercept, slope = model.intercept_, model.coef_


# Define the intensity function
def intensity_function(t):
    """
    Defines the intensity function for an inhomogeneous Poisson process.

    Parameters:
    - t (float): Time parameter.

    Returns:
    - float: The intensity of the process at the given time.
    """
    return slope * t + intercept


## Simulation

def simulate_inhomogeneous_poisson_process(total_time, time_increment):
    """
    Simulates an inhomogeneous Poisson process over a specified time period.

    Parameters:
    - total_time (float): Total time duration of the simulation in years.
    - time_increment (float): Time increment for the simulation in years.

    Returns:
    - t (numpy.ndarray): Array of time points.
    - events (numpy.ndarray): Array representing the number of events at each time point.

    Note:
    The Poisson process is a stochastic model that describes the number of events
    occurring in fixed intervals of time or space. The `rate` parameter represents
    the average rate of events per unit time, and the simulation is performed
    using the provided time increment over the specified total time.
    """
    t = np.arange(2015, 2015 + total_time, time_increment)
    num_increments = len(t)
    events = np.zeros(num_increments)

    for i in range(1, num_increments):
        intensity = intensity_function(t[i - 1])
        num_events = np.random.poisson(intensity * time_increment)
        events[i] = num_events

    return t, events

# Parameters for simulation
simulation_time = 50  # Total time in years
time_increment = 1  # Time increment for simulation in years

# Simulate Poisson process
x2, event_counts = simulate_inhomogeneous_poisson_process(simulation_time, time_increment)

# Adjust the first element based on the intensity at the initial time
initial_intensity = intensity_function(x2[0])
event_counts[0] = np.random.poisson(initial_intensity * time_increment)


## Combine the two graphs

# Define Variables
x1 = x1.unique()[::-1]
X = np.concatenate((x1, x2))

y1 = counts
y2 = event_counts
Y = np.concatenate((y1, y2))

# Create a larger figure
plt.figure(figsize=(18, 12))

# Plotting the results with bars
plt.bar(X, Y, edgecolor='black', color="#E6A000", width=time_increment, align='edge')
plt.xlabel('Time (Years)', fontsize=14)
plt.ylabel('Number of Catastrophes', fontsize=14)
plt.title('Simulation of the Number of Natural Disasters in Australia after 2014, Combined with Original Dataset', fontsize=20)
plt.grid(axis='y')  # Add grid lines on the y-axis for better readability
plt.show()
