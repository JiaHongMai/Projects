# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:51:46 2023

@author: Reydarz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
# Assuming you have the dataset loaded into a variable named 'data'
data = "auscathist.xlsx"

data = pd.read_excel(data)

# Extract the 'NormCost2014' column as the target distribution
target_distribution = data['NormCost2014']

# Define the proposal distribution (Poisson distribution)
def proposal_distribution(theta):
    return np.random.poisson(theta)

# Define the target distribution (using the 'NormCost2014' column)
def target_distribution_prob(theta):
    # Assuming a normal distribution for simplicity
    mu = np.mean(target_distribution)
    sigma = np.std(target_distribution)
    return np.exp(-0.5 * ((theta - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations, initial_theta):
    samples = [initial_theta]

    for _ in range(iterations):
        # Propose a new sample from the proposal distribution
        theta_proposed = proposal_distribution(samples[-1])

        # Calculate acceptance ratio
        alpha = min(1, target_distribution_prob(theta_proposed) / target_distribution_prob(samples[-1]))

        # Accept or reject the proposed sample
        if np.random.rand() < alpha:
            samples.append(theta_proposed)
        else:
            samples.append(samples[-1])

    return np.array(samples)

# Set the number of iterations and initial value
iterations = 10000
initial_value = np.mean(target_distribution)

# Run Metropolis-Hastings algorithm
samples = metropolis_hastings(iterations, initial_value)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(samples, label='Metropolis-Hastings Samples')
plt.axhline(np.mean(target_distribution), color='red', linestyle='--', label='True Mean')
plt.legend()
plt.title('Metropolis-Hastings Simulation of Insurance Losses with Poisson Proposal Distribution')
plt.xlabel('Iterations')
plt.ylabel('Sampled Values')
plt.show()
