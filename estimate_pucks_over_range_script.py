from pucks_to_penetrate_cryosphere import evaluate_number_of_pucks_on_arbitrary_europa
# from test_mp import *
import pandas as pd
import multiprocess as mp
import numpy as np

import os
from datetime import datetime
import time

if __name__ == "__main__":
    # Get the current date and time, formatted as YYYY-MM-DD_HH-MM-SS
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory with the current date and time as the name
    os.makedirs(current_time, exist_ok=True)

    # Change the current working directory to the newly created directory
    os.chdir(current_time)

    # Verify the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    default_T_u = 104 #K
    default_T_l = 230 #K
    default_T_conv = 251.6 #K
    default_D_cond = 10.4e3 #m
    default_D_phi = 3.2e3 #m
    default_D_conv = 5.8e3 #m
    default_eta_vac = 0.1 #np.arrange(0, 0.3, 0.1)
    default_rho_salt = 1e-5 #np.linspace(1e-3, 4.2e-2, 10)
    default_H = 0.75 # Surface roughness hurst coefficient
    default_sigma_ref = 0.2 # Surface roughness at reference wavelength (at 1m)
    default_file_suffix = ''

    default_T_melt = 273.13 #K

    dictionary_list = []
    # Construct a dataframe of Europas 
    # dictionary_list.append({
    #     "eta_vac": default_eta_vac,  # porosity
    #     "rho_salt": default_rho_salt,  # salt fraction

    #     "T_u": default_T_u,  #K
    #     "T_l": default_T_l,  #K
    #     "T_conv": default_T_conv,  #K
    #     "D_cond": default_D_cond,  #m
    #     "D_phi": default_D_phi,  #m
    #     "D_conv": default_D_conv,  #m
    #     "H": default_H,  
    #     "sigma_ref": default_sigma_ref, 
    #     "file_suffix": 'test' 
    # })

    # with a range of salts and vacuum fractions
    for eta_vac in [0.1, 0.3]:
        for rho_salt in [0, 1e-6, 1e-5, 1e-4, 1e-3, 4.2e-2]:
            gen_d_file_suffix = f"_porosity{eta_vac}_saltfraction{rho_salt:.1e}".replace('.', 'p') 
            dictionary_list.append({
                "eta_vac": eta_vac,  # porosity
                "rho_salt": rho_salt,  # salt fraction

                "T_u": default_T_u,  #K
                "T_l": default_T_l,  #K
                "T_conv": default_T_conv,  #K
                "D_cond": default_D_cond,  #m
                "D_phi": default_D_phi,  #m
                "D_conv": default_D_conv,  #m
                "H": default_H,  
                "sigma_ref": default_sigma_ref, 
                "file_suffix": gen_d_file_suffix 
            })

    # With a limit of size Europas do the same
    max_D_cond = 10.4e3 + 5.8e3
    max_D_conv = 5.8e3 + 6.3e3
    for eta_vac in [0.1, 0.3]:
        for rho_salt in [1e-3, 4.2e-2]:
            gen_d_file_suffix = f"_{max_D_conv}_{max_D_conv}_porosity{eta_vac}_saltfraction{rho_salt:.1e}".replace('.', 'p') 
            dictionary_list.append({
                "eta_vac": eta_vac,  # porosity
                "rho_salt": rho_salt,  # salt fraction
                "D_cond": max_D_cond,  #m
                "D_phi": 0.43*max_D_cond,  #m
                "D_conv": max_D_conv,#m

                "T_u": default_T_u,  #K
                "T_l": default_T_l,  #K
                "T_conv": default_T_conv,  #K

                "H": default_H,  
                "sigma_ref": default_sigma_ref, 

                "file_suffix": gen_d_file_suffix
            })

    df = pd.DataFrame(dictionary_list)
    df['idx'] = df.index

    total_jobs = len(df)

    print(f'Generated {total_jobs} Europa scenarios. Saved scenarios to number_of_pucks_estimates.csv')
    df.to_csv('number_of_pucks_estimates.csv')
    
    print(f'Launching {mp.cpu_count()+1} worker processes...')
    # handle raised errors
    def handle_error(error):
        print(error, flush=True)

    def monitor_progress(progress, lock, total_jobs, start_time):
        def pp_elapsed_time(elapsed_time):
            days, remainder = divmod(elapsed_time, 86400)  # 86400 seconds in a day
            hours, remainder = divmod(remainder, 3600)     # 3600 seconds in an hour
            minutes, seconds = divmod(remainder, 60)       # 60 seconds in a minute

            return f"{int(days)}D {int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        last_progress_value = 0
        while last_progress_value < total_jobs:
            with lock:
                last_progress_value = progress.value
            elapsed_time = time.time() - start_time
            running_workers = mp.cpu_count()+1
            
            print(f"\rTotal jobs: {total_jobs}, "
                f"Jobs completed: {last_progress_value}, "
                f"Running workers: {running_workers}, "
                f"Elapsed time: " + pp_elapsed_time(elapsed_time) + ".", end="", flush=True)
            time.sleep(1)  # Update progress every 1 second

        # Final update after all jobs are completed
        elapsed_time = time.time() - start_time
        print(f"\nAll jobs completed in " + pp_elapsed_time(elapsed_time) + ".", flush=True)

    # Function to apply to apply to each row
    def apply_function_to_row(row, progress, lock):
        # Replace this with the actual processing logic
        T_u = row["T_u"] #K
        T_l = row["T_l"] #K
        T_melt = 273.13 #K
        T_conv = row["T_conv"] #K
        D_cond = row["D_cond"] #m
        D_phi = row["D_phi"] #m
        D_conv = row["D_conv"] #m
        eta_vac = row["eta_vac"] # porosity
        rho_salt = row["rho_salt"] # salt fraction
        H = row["H"]
        sigma_ref = row["sigma_ref"]
        file_suffix = row["file_suffix"]
        delta_d = 10 #m

        try:
            results = evaluate_number_of_pucks_on_arbitrary_europa(
                        T_u, T_l, T_melt, T_conv, 
                        D_cond, D_phi, 
                        eta_vac, rho_salt, 
                        D_conv, delta_d, 
                        H, sigma_ref,
                        file_suffix)
        finally:
            # Increment completed job count
            with lock:  # Use a lock to prevent race conditions
                progress.value += 1
        
        return results

    # Start time tracking
    start_time = time.time()

    # Create a manager object to share values
    with mp.Manager() as manager: 
        lock = manager.Lock()
        progress = manager.Value('i', 0)  # Track the number of completed jobs
        
        # Create a pool of worker processes
        with mp.Pool(processes=mp.cpu_count()+1) as pool:
            # Create a monitoring thread in the pool
            monitor_result = pool.apply_async(monitor_progress, 
                args=(progress, lock, total_jobs, start_time))

            # Apply the function to each row in parallel using pool.starmap
            results = pool.starmap_async(apply_function_to_row, 
                [(row, progress, lock) for _, row in df.iterrows()],
                error_callback=handle_error)
            
            monitor_result.wait()

    uhf_pucks, uhf_pucks_cond, uhf_pucks_conv, hf_pucks, hf_pucks_cond, hf_pucks_conv  = list(zip(*results.get()))
    # Convert the results back to a DataFrame or any other desired format
    df['N_UHF'] = uhf_pucks
    df['N_UHF Conductive'] = uhf_pucks_cond
    df['N_UHF Convective'] = uhf_pucks_conv
    df['N_HF'] = hf_pucks
    df['N_HF Conductive'] = hf_pucks_cond
    df['N_HF Convective'] = hf_pucks_conv

    df.to_csv('number_of_pucks_estimates.csv')