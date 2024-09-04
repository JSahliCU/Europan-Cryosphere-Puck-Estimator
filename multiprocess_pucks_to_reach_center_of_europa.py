# from test_mp import *
import pandas as pd
import multiprocess as mp
import numpy as np

from pucks_to_penetrate_cryosphere import evaluate_number_of_pucks_on_arbitrary_europa

# Set the number of rows you want in the DataFrame
num_rows = 100  # You can adjust this number

# Create a DataFrame with three columns of random data
df = pd.read_csv('SMHowell-Europa-22aa171/Europas.csv', index_col=0, nrows=5)

# Example function to apply to each row
def process_row(row):
    # Replace this with the actual processing logic
    T_u = row["T_s"] #K
    T_l = row["T_cond_base"] #K
    T_melt = 273.13 #K
    T_conv = row["T_c"] #K
    D_cond = row["D_cond"] #m
    D_phi = row["D_brittle"] #m
    D_conv = row["D_conv"] #m
    eta_vac = row["phi"] # porosity
    rho_salt = row["f_s"] # salt fraction
    delta_d = 10 #m

    return evaluate_number_of_pucks_on_arbitrary_europa(T_u, T_l, T_melt, T_conv, D_cond, D_phi, eta_vac, rho_salt, D_conv, delta_d)

# Function to apply the process_row function to each row
def apply_function_to_row(row):
    return process_row(row)

with mp.Pool(processes=5) as pool:
    # Apply the function to each row of the DataFrame using the pool of worker processes
    results = pool.map(apply_function_to_row, [row for _, row in df.iterrows()])

n_hf, n_uhf  = list(zip(*results))
# Convert the results back to a DataFrame or any other desired format
df['N_UHF'] = n_uhf
df['N_HF'] = n_hf

print(df)