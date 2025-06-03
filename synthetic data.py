# Generation of synthetic data from the paper titled "Prediction of Sugar yields during hydrolysis of lignocellulosic biomass using artificial neural network modeling"
# Since the complete data wasnt avaiable, synthetic data was generated using the data from table 4 in the paper.

import pandas as pd 
import numpy as np

particle_size_codes = {
    "<0.5 mm": 0,
    "0.5 - 1.0 mm": 1,
    ">1.0 mm": 2,
    "Mixed": 3
}

# define the conditions in table as 'conditions' ( biomass, particle size, glucose range, xylose range)

conditions = [
    [10, "<0.5 mm", (21.1, 41.7), (7.2, 15.9)],
    [10, "0.5 - 1.0 mm", (21.4, 43.4), (7.5, 15.5)],
    [10, ">1.0 mm", (19.9, 49.3), (7.06, 15.3)],
    [15, "<0.5 mm", (26.4, 55.3), (12.6, 21.05)],
    [15, "0.5 - 1.0 mm", (23.4, 60.1), (11.6, 22.3)],
    [15, ">1.0 mm", (24.7, 61.6), (13.15, 22.8)],
    [18, "<0.5 mm", (27.9, 61.4), (12.3, 24.2)],
    [18, "0.5 - 1.0 mm", (26.6, 67.0), (14.7, 25.01)],
    [18, ">1.0 mm", (26.5, 73.4), (14.2, 24.9)],
    [10, "Mixed", (20.3, 44.7), (7.1, 12.9)],
    [15, "Mixed", (25.4, 60.1), (12.9, 21.8)],
    [18, "Mixed", (26.0, 71.6), (15.01, 24.3)]
]

# Time range for all
time_range = (4,48) # written separetly as the time range is constant for all the experimental conditions 
samples_per_condition = 25

# An empty list was created to store the synthetic data that will be generated
synthetic_data = []

np.random.seed(42) # this allows us to get the same result or data when someone runs the code anytime

for condition in conditions:
    # biomass = 10, size_label = <0.5 mm, glucose_range = (21.1, 41.7), xylose_range = (7.2, 15.9)
    biomass, size_label, glucose_range, xylose_range = condition 
    size_code = particle_size_codes[size_label]
    
    for _ in range(samples_per_condition):
        time = np.random.uniform(*time_range)
        glucose = np.random.uniform(*glucose_range)
        xylose = np.random.uniform(*xylose_range)
        
        synthetic_data.append([biomass, size_code, time, glucose, xylose])
        

# Conversion of the newly generated syntehtic data into a DataFrame 
column = ['biomass_loading', 'particle_size', 'time', 'glucose', 'xylose']
df = pd.DataFrame(synthetic_data, columns=column)

# to check if the data is formatted properly
# gives us the first ten rows of the synthetic data 
print(df.head(10)) 

# save the dataframe in a .csv file in a desried folder 
df.to_csv(r"C:\Users\akash singh\Desktop\thesis\synthetic_data.csv", index=False)




