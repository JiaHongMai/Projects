# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# Assuming you have the dataset loaded into a variable named 'data'
data = r"C:\Users\Reydarz\OneDrive\Projects\CASdatasets\data\auscathist.xlsx"

df = pd.read_excel(data)

#%%Histogram

x1 = df["Year"]

# Create a larger figure
plt.figure(figsize=(18,12))

hist = plt.hist(x1,bins=len(set(x1)), edgecolor='black', color="#E6A000")
plt.ylabel('Number of catastrophe',fontsize=10)
plt.title('Number of natural disasters per year in Australia', fontsize=20)

y1, _ = np.histogram(x1,bins=len(set(x1)))


# Set ticks to show every increment on both x and y axes
highest_count = int(max(hist[0]))

plt.xticks(range(min(x1), max(x1)),fontsize=10,rotation=35)
plt.yticks(range(0, highest_count),fontsize=5)

# Fit a trendline (linear regression) to the counts
counts, bins, _ = hist

coefficients = np.polyfit(bins[:-1], counts, 1)
trendline = np.poly1d(coefficients)
plt.plot(bins[:-1], trendline(bins[:-1]), color='red', linestyle='--', linewidth=2, label='Trendline')
plt.show()




#Added Period column
def map_to_period(year):
    return f"{(year // 5) * 5} - {(year // 5) * 5 + 4}"

df['Period'] = df['Year'].apply(map_to_period)

#Estimate the mean of disasters per period

dis_lambda = df.groupby('Year').size().mean()



##Simulation
def simulate_poisson_process(rate, total_time, time_increment):
    num_increments = int(total_time / time_increment)
    t = np.arange(2015, 2015+total_time, time_increment)
    events = np.zeros(num_increments)
    for i in range(num_increments):
        num_events = np.random.poisson(rate * time_increment)
        events[i] = num_events
    return t, events

# Parameters
poisson_rate = dis_lambda
simulation_time = 50  # Total time in years
time_increment = 1  # Time increment for simulation in years

# Simulate Poisson process
x2, event_counts = simulate_poisson_process(poisson_rate, simulation_time, time_increment)

# Create a larger figure
plt.figure(figsize=(18, 12))

# Plotting the results with bars
plt.bar(x2, event_counts, edgecolor='black', color="#E6A000", width=time_increment, align='edge')
plt.xlabel('Time (Years)', fontsize=14)
plt.ylabel('Number of Catastrophes', fontsize=14)
plt.title('Number of Natural Disasters over Time', fontsize=20)
plt.grid(axis='y')  # Add grid lines on the y-axis for better readability
plt.show()



## Combine the two graphs

#Define Variables
x1 = x1.unique()[::-1]
X = np.concatenate((x1, x2))

y2 = event_counts
Y = np.concatenate((y1, y2))

# Create a larger figure
plt.figure(figsize=(18, 12))

# Plotting the results with bars
plt.bar(X, Y, edgecolor='black', color="#E6A000", width=time_increment, align='edge')
plt.xlabel('Time (Years)', fontsize=14)
plt.ylabel('Number of Catastrophes', fontsize=14)
plt.title('Number of Natural Disasters over Time', fontsize=20)
plt.grid(axis='y')  # Add grid lines on the y-axis for better readability
plt.show()
plt.title('Metropolis-Hastings Simulation of Insurance Losses with Poisson Proposal Distribution')
plt.xlabel('Iterations')
plt.ylabel('Sampled Values')
plt.show()
