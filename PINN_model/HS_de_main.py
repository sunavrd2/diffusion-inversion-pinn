import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import pickle

from early_stop_callback_two_nets import Callback_EarlyStopping
from adaptive_activations import AdaptiveTanh, AdaptiveSwish

from PINN_de import Diffusion_PDE

# List all visible GPUs 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print("Could not set memory growth:", e)
else:
    print("No GPUs found.")


current_dir = os.getcwd()

##------------------------------------------------------------------------------------------------------------------------##
## LOAD THE DATA FOR TRAINING
#################################################################

#################################################################

#################################################################

##------------------------------------------------------------------------------------------------------------------------##

data_all = np.load(os.path.join(current_dir,"..","data/HS_log_profiles.npy"))
data_all = data_all * 0.805             # scaling is applied to match with the gravimetric absorbed water content at lower boundary 
# print(data_all.shape)


pixel_size = 31.63      # micrometers um   - determined from the X-ray setting/radiograph metadata
x_data_all = np.arange(data_all.shape[1])*pixel_size/1000 #in milimeters

# print(x_data.shape)

# Scaled the X_data to range [-1,1]
x_data_scaled_all = -1 + 2 * x_data_all/np.max(x_data_all)

#Let's load all the t_data regardless of the holdout and also lbc value, lbc data and t_data max should be independant of the hold out so we do it outside the loop
t_data_all = np.load(os.path.join(current_dir,"..","data/HS_difference_times.npy"))
t_data_all = t_data_all
t_data_max_all = np.max(t_data_all)
#print("t_data_max:", t_data_max)
t_data_scaled_all = t_data_all/np.max(t_data_all)

# Check shape
print("Water Content data shape:", data_all.shape)
print("Time data shape:", t_data_all.shape)

#################################################################

#################################################################

#################################################################


####### TRAIN ONLY USING 80% DATA SO SELECT ACCORDINGLY ##########
### Let's only select 80% of data for training

select_idx = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24]

t_data_scaled_all = t_data_scaled_all[select_idx]

data_all = data_all[select_idx,:]

#################################################################

#################################################################

#################################################################

#################################################################


#scaling factor for mass atteniation (mu/rho) at 80KeV - this is calculated from the NIST attenuation plots for water. 
scaling_factor = -5.44366 #gm/cm3               # see the procedure in the supplementary info for details. 
s_data_all = scaling_factor * data_all.T

time_lbc_start = 5.0/t_data_max_all                 ########Note that we only reliably know of a fixed lbc from this time onwards
lbc_value = np.mean(s_data_all[0,3:])
print("lbc value:", lbc_value)
lbc_std = np.std(s_data_all[0,3:])


##------------------------------------------------------------------------------------------------------------------------##
## SPECIFY SAVE DIRECTORY
##------------------------------------------------------------------------------------------------------------------------##

save_dir = os.path.join(current_dir,"HS_outputs")
os.makedirs(save_dir, exist_ok=True)

##------------------------------------------------------------------------------------------------------------------------##
## RUN 750 ENSEMBLES
##------------------------------------------------------------------------------------------------------------------------##

count = 0 # How many times we've updated the sums

# Initialize running sums to None
if count == 0:
    u_hat_sum = None
    u_hat_sq_sum = None
    d_hat_sum = None
    d_hat_sq_sum = None

else:
    u_hat_mean = np.load(os.path.join(save_dir,f'u_hat_mean_{count}.npy'))
    u_hat_std = np.load(os.path.join(save_dir,f'u_hat_std_{count}.npy'))

    d_hat_mean = np.load(os.path.join(save_dir,f'd_hat_mean_{count}.npy'))
    d_hat_std = np.load(os.path.join(save_dir,f'd_hat_std_{count}.npy'))

    u_hat_sum = u_hat_mean * count
    u_hat_sq_sum = (np.square(u_hat_std) + np.square(u_hat_mean)) * count

    d_hat_sum = d_hat_mean * count
    d_hat_sq_sum = (np.square(d_hat_std) + np.square(d_hat_mean)) * count
    
initial_count = count 

##------------------------------------------------------------------------------------------------------------------------##
## PROCESS THE LOADED DATA - MAKE INTERPOLATION GRIDS ETC.
##------------------------------------------------------------------------------------------------------------------------##

x_data_msh, t_data_msh = np.meshgrid(x_data_scaled_all,t_data_scaled_all)

x_data = x_data_msh.flatten()
t_data = t_data_msh.flatten()
s_data = s_data_all.T.flatten()

##--------------------------------------------------------------------------------------------------------------------------------##
## VARIANCE AND STDEV VALUES FOR RANDOM NOISE SAMPLING - these are obtained from optimal training using grid search
##--------------------------------------------------------------------------------------------------------------------------------##  
# Note the idea is scaling all the loss terms by same factors is essentially minimizing the same objective

# These are starting weights derived from the grid search approach.
residual_weight = 1.0
data_weight = 5.0 
ic_weight = 2.0 
lbc_weight = 2.0 
ubc_weight = 2.0 
theta_weight = 0.1 
phi_weight = 0.1 


eff_wt_res = residual_weight
eff_wt_data = data_weight 
eff_wt_init = ic_weight
eff_wt_lbc = lbc_weight
eff_wt_ubc = ubc_weight
eff_wt_theta = theta_weight
eff_wt_phi = phi_weight


time0 = time.time()    

init_params = {
't_lbc_start' : time_lbc_start,

'eff_wt_res' : eff_wt_res,
'eff_wt_data' : eff_wt_data,
'eff_wt_init' : eff_wt_init,
'eff_wt_lbc' :eff_wt_lbc,
'eff_wt_ubc' : eff_wt_ubc,
'eff_wt_theta' : eff_wt_theta,
'eff_wt_phi' : eff_wt_phi
}


# Run 750 ensembles
for ens in range(1,751):

    # Initialize the randomized PINN instance
    PINN = Diffusion_PDE(x_data, t_data, s_data, time_lbc_start, lbc_value, seed_value = 123 * ens , resample_freq = 100, **init_params)
    #Train the PINN model
    loss_logs, best_logs = PINN.train()

    #Calculate the Residul and Data loss on the holdout set. 

    #save all the relevant parameters

    np.save(os.path.join(save_dir,f'HS_best_{ens}.npy'),best_logs)
    np.save(os.path.join(save_dir,f'HS_logs_{ens}.npy'),loss_logs)

    #Let's infer for the entire domain
    u_hat, d_hat = PINN.infer_in_batches()
    
    np.save(os.path.join(save_dir, f"u_hat_{ens}.npy"), u_hat)
    np.save(os.path.join(save_dir, f"d_hat_{ens}.npy"), d_hat)

    if u_hat_sum is None:
        u_hat_sum = np.zeros_like(u_hat)
        u_hat_sq_sum = np.zeros_like(u_hat)
        d_hat_sum = np.zeros_like(d_hat)
        d_hat_sq_sum = np.zeros_like(d_hat)


    # Update running sums
    u_hat_sum += u_hat
    u_hat_sq_sum += u_hat**2

    d_hat_sum += d_hat
    d_hat_sq_sum += d_hat**2

    count += 1


    # Save every 25 iterations 
    if (ens) % 25 == 0:
        # Compute means
        u_hat_mean = u_hat_sum / count
        d_hat_mean = d_hat_sum / count

        # Compute variances = E(X^2) - [E(X)]^2
        u_hat_var = (u_hat_sq_sum / count) - (u_hat_mean ** 2)
        d_hat_var = (d_hat_sq_sum / count) - (d_hat_mean ** 2)

        # Standard deviation
        u_hat_std = np.sqrt(u_hat_var)
        d_hat_std = np.sqrt(d_hat_var)

        # Save them to disk
        np.save(os.path.join(save_dir, f"u_hat_mean_{ens}.npy"), u_hat_mean)
        np.save(os.path.join(save_dir, f"u_hat_std_{ens}.npy"), u_hat_std)
        np.save(os.path.join(save_dir, f"d_hat_mean_{ens}.npy"), d_hat_mean)
        np.save(os.path.join(save_dir, f"d_hat_std_{ens}.npy"), d_hat_std)

        print(f"Saved running mean/std at ensemble: {ens}")

    print(f"!!!!!!!!!!!Completed ensemble: {ens}!!!!!!!!!!!!!!")

        



