import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm

from early_stop_callback_two_nets import Callback_EarlyStopping
from adaptive_activations import AdaptiveTanh, AdaptiveSwish

class Diffusion_PDE(tf.keras.Model):
    def __init__(self, x_data, t_data, s_data, time_lbc_start, lbc_value, seed_value, resample_freq = 100, **kwargs):
        super().__init__()
        self.x_min = -1.0
        self.x_max = 1.0
        self.t_min = 0.0
        self.t_max = 1.0
        
        self.data_x = x_data.reshape(-1,1)
        self.data_t = t_data.reshape(-1,1)
        
        self.data_S = s_data.reshape(-1,1)
        self.data_size = s_data.shape[0]
        self.lbc_value = lbc_value # lbc_value: steady-state saturation at lower ROI boundary (post-stabilization)

        self.t_lower_start = time_lbc_start
        
        self.init_t = 0.00

        # Unpack additional tensors
        
        self.t_lbc_start    = kwargs.get('t_lbc_start', None)

        self.wt_res         = kwargs.get('eff_wt_res', None)
        self.wt_data        = kwargs.get('eff_wt_data', None)
        self.wt_init        = kwargs.get('eff_wt_init', None)
        self.wt_lbc         = kwargs.get('eff_wt_lbc', None)
        self.wt_ubc         = kwargs.get('eff_wt_ubc', None)
        self.wt_theta       = kwargs.get('eff_wt_theta', None)
        self.wt_phi         = kwargs.get('eff_wt_phi', None)

        # Setup idx tensors for shuffling and sampling
        self.idx_data = np.arange(self.data_x.shape[0], dtype=np.int32)
        

    # Set the network parameters
       
        self.num_input = 2
        self.num_output = 1
        self.num_layers = 10
        self.num_neurons = 40
        self.num_layers_coeff = 10
        self.num_neurons_coeff = 40
        self.activ = 'swish'             # Activation function
        self.kernel_init = 'glorot_normal'   # Layer weights initializer
        self.b_init = 'zeros'           # Bias initializer


        #Set seeds for repeatibility        
        self.seed = seed_value
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
            
        # Set the optimizers and learning rate for training
        initial_learning_rate = 1e-3

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=2000,  # Number of steps after which the learning rate decays
            decay_rate=0.5,    # The decay rate
            staircase=True      # If True, decay the learning rate at discrete intervals
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.epoch = int(50e3)
        self.mntr = 50
        self.resample_freq = resample_freq

        
        
        self.adapt_res = tf.constant(self.wt_res, dtype=tf.float32)
        self.adapt_data = tf.constant(self.wt_data, dtype=tf.float32)
        self.adapt_ic = tf.constant(self.wt_init, dtype=tf.float32)
        self.adapt_lbc = tf.constant(self.wt_lbc, dtype=tf.float32)
        self.adapt_ubc = tf.constant(self.wt_ubc, dtype=tf.float32)  

        self.adapt_dnn_reg = tf.constant(self.wt_theta, dtype=tf.float32)
        self.adapt_coef_reg = tf.constant(self.wt_phi, dtype=tf.float32)   

        self.alpha = tf.constant(0.1, dtype=tf.float32)  #this aplha is the rate for the weight annealing 
        
        
        #Logs for adaptive weights
        logs_adaptive = {'epoch':[], 'max_residual': [], 'adaptive_ubc':[], 'adaptive_lbc': [], 'adaptive_init':[]}
 
        self.dnn = self.build_network(self.num_input, self.num_output, self.num_layers, self.num_neurons)
        self.coeff_net = self.build_network(1, self.num_output, self.num_layers_coeff, self.num_neurons_coeff)
        
        self.params = self.dnn.trainable_variables + self.coeff_net.trainable_variables
        
        # Track loss
        self.ep_log = []
        self.total_loss_log = []
        self.residual_log = []
        self.lb_loss_log = []
        self.ub_loss_log = []
        self.init_loss_log = []
        self.data_loss_log=[]
        
        self.weights_list_res = []
        self.weights_list_data = []
        self.weights_list_ic = []
        self.weights_list_lbc = []
        self.weights_list_ubc = []
        self.weight_list_dnn = []
        self.weight_list_coef = []

        self.best_log = {
            'epoch': [],
            'loss_weights_rdilu' : [],
            'losses_rdilu': []}
        


    def build_network(self, num_input, num_output, num_layers, num_neurons):
            network = tf.keras.Sequential()
            
            # Input Layer
            network.add(tf.keras.layers.InputLayer(input_shape=(num_input,)))
            
            # Hidden Layers with per-neuron adaptive Swish activation
            for _ in range(num_layers):
                # Use Glorot Normal for weights and zeros for bias
                network.add(tf.keras.layers.Dense(num_neurons, use_bias=True,
                                                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                bias_initializer=tf.zeros_initializer()))
                
                # After each Dense layer, apply neuron-specific adaptive Swish activation
                network.add(AdaptiveSwish(num_neurons))
            
            # Output Layer (no activation function here)
            network.add(tf.keras.layers.Dense(num_output))
            
            return network
        
        
    def call(self, inputs):  # Inputs are tensors of size (batch_size, 2)
        return self.dnn(inputs)
    
    
    def call_coeff_net(self, inputs):  # Inputs are tensors of size (batch_size, 2)
        return self.coeff_net(inputs)
    
    def predict(self, x, t): 
        u_hat = self.call(tf.concat([x, t], axis=-1))
        return u_hat
    
    
    def predict_coeff(self,s):
        d_coeff = self.call_coeff_net( s)
        return tf.exp(d_coeff)
    

    def PDE(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            u = self.predict(x, t)
            d = self.predict_coeff(u)
            u_x = tape.gradient(u, x)
            d_ux = d * u_x
    
        d_d_ux = tape.gradient(d_ux, x)
        u_t = tape.gradient(u, t)     
               
        
        residual = u_t - d_d_ux
        
        del tape

        return u_x, d, residual
    
    
    def compute_losses(self, x_res, t_res, x_upper, t_upper, x_lower, t_lower,  x_dat, t_dat, u_dat,  x_init, t_init, u_init):

        _, _, pde_loss = self.PDE(x_res, t_res)
        ubc_loss, _,_ = self.PDE(x_upper, t_upper)
        data_pred = self.predict(x_dat, t_dat)
        lbc_loss = self.predict(x_lower, t_lower)
        ic_pred = self.predict(x_init, t_init)
        
        pde_loss = tf.reduce_mean(tf.square(pde_loss))
        ubc_loss = tf.reduce_mean(tf.square(ubc_loss))
        lbc_loss = tf.reduce_mean(tf.square(lbc_loss- (self.lbc_value*tf.ones_like(lbc_loss))))
        data_loss = tf.reduce_mean(tf.square(u_dat - data_pred))
        ic_loss = tf.reduce_mean(tf.square(ic_pred))

        #Lets add two reguralizer terms for the two nets
        dnn_reg = tf.add_n([tf.reduce_mean(tf.square(w)) for w in self.dnn.trainable_variables])
        coeff_reg = tf.add_n([tf.reduce_mean(tf.square(w)) for w in self.coeff_net.trainable_variables])
        return pde_loss, ubc_loss, lbc_loss, data_loss, ic_loss, dnn_reg, coeff_reg
        
      
    @tf.function
    def train_step(self, x_res, t_res, x_upper, t_upper, x_lower, t_lower,  x_dat, t_dat, u_dat, x_init, t_init, u_init):

        with tf.GradientTape(persistent=True) as tape:
            pde_loss, ubc_loss, lbc_loss, data_loss, ic_loss, dnn_reg, coeff_net_reg = self.compute_losses(x_res, t_res, x_upper, t_upper, x_lower, t_lower,  x_dat, t_dat, u_dat,  x_init, t_init, u_init)
                             
            total_loss = self.adapt_res * pde_loss  +  self.adapt_data * data_loss +  self.adapt_ic * ic_loss + self.adapt_lbc * lbc_loss #+ 10.0 * self.adapt_ubc * ubc_loss
        
        gradients = tape.gradient(total_loss,self.params)

        self.optimizer.apply_gradients(zip(gradients, self.params))
        
        del tape

        return total_loss, pde_loss, data_loss, ubc_loss, lbc_loss, ic_loss



    @tf.function
    def loss_grad(self, x_res, t_res, x_upper, t_upper, x_dat, t_dat, u_dat, x_lower, t_lower, x_init, t_init, u_init):

        with tf.GradientTape(persistent=True) as tape:
            pde_loss, ubc_loss, lbc_loss, data_loss, ic_loss, dnn_reg, coeff_net_reg = self.compute_losses(x_res, t_res, x_upper, t_upper, x_lower, t_lower,  x_dat, t_dat, u_dat,  x_init, t_init, u_init)
            
            pde_grad = tape.gradient(pde_loss,self.params)
            lbc_grad = tape.gradient(lbc_loss,self.params)
            ubc_grad = tape.gradient(ubc_loss,self.params)
            data_grad = tape.gradient(data_loss, self.params)
            ic_grad = tape.gradient(ic_loss,self.params)     
            dnn_reg_grad = tape.gradient(dnn_reg,self.params)
            coeff_net_reg_grad = tape.gradient(coeff_net_reg,self.params)
                 
        del tape       
        
        return pde_grad, ubc_grad, lbc_grad, data_grad, ic_grad, dnn_reg_grad, coeff_net_reg_grad


    ###Dynamically Update the Weights for Loss Terms
    def update_weights (self,pde_grad, ubc_grad, lbc_grad, data_grad, ic_grad, dnn_reg_grad, coeff_net_reg_grad):
              
        max_pde_grad = tf.reduce_max([tf.reduce_max(tf.abs(g)) for g in [k for k in pde_grad if k is not None]])
        mean_lbc_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in [k for k in lbc_grad if k is not None]])
        mean_ubc_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in [k for k in ubc_grad if k is not None]])
        mean_data_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in [k for k in data_grad if k is not None]])
        mean_ic_grad = tf.reduce_mean([tf.reduce_mean(tf.abs(g)) for g in [k for k in ic_grad if k is not None]])

        ######Note : for first iter, self.adapt_ubc/lbc/init are initialized as obtained from grid search
        #Here we update the adaptive weights based on the scaling to rebalance wrt PDE loss.       
        self.adapt_lbc = (1-self.alpha) * self.adapt_lbc + self.alpha * ( max_pde_grad/(self.adapt_lbc * mean_lbc_grad))
        self.adapt_ubc = (1-self.alpha) * self.adapt_ubc + self.alpha * (max_pde_grad/(self.adapt_ubc * mean_ubc_grad))
        self.adapt_data = (1-self.alpha) *self.adapt_data + self.alpha * (max_pde_grad/(self.adapt_data * mean_data_grad))
        self.adapt_ic = (1-self.alpha) * self.adapt_ic + self.alpha * (max_pde_grad/(self.adapt_ic * mean_ic_grad)) 



    def random_sample(self, lb, ub, batch_size):
        return tf.convert_to_tensor(np.random.uniform(lb, ub, batch_size).reshape(-1,1),dtype=tf.float32)


    def get_train_dataset(self, batch_size_domain=20000, batch_size_upper=100, batch_size_lower=1000, batch_size_initial=1000):
        x_res = self.random_sample(self.x_min, self.x_max, batch_size_domain)
        t_res = self.random_sample(self.t_min, self.t_max, batch_size_domain)
        
        idx_data = np.random.permutation(self.idx_data)

        x_dat = tf.convert_to_tensor(self.data_x[idx_data], dtype=tf.float32)
        t_dat = tf.convert_to_tensor(self.data_t[idx_data], dtype=tf.float32)
        u_dat = tf.convert_to_tensor(self.data_S[idx_data], dtype=tf.float32)
        
        x_init = self.random_sample(self.x_min, self.x_max, batch_size_initial)
        t_init = tf.ones_like(x_init) * self.init_t
        u_init = tf.zeros_like(x_init)
        
        t_upper = self.random_sample(self.t_min, self.t_max, batch_size_upper)
        x_upper = tf.ones_like(t_upper)
        
        t_lower = self.random_sample(self.t_lbc_start, 2 * self.t_max, batch_size_lower)     # start from when lbc has stabilized; sample beyond t_max to enforce BC after stabilization         
        x_lower = -1.0 * tf.ones_like(t_lower)
           
        return x_res, t_res, x_upper, t_upper, x_lower, t_lower, x_dat, t_dat, u_dat, x_init, t_init, u_init


    def train(self):
        t0 = time.time()
        x_res, t_res, x_upper, t_upper, x_lower, t_lower, x_dat, t_dat, u_dat, x_init, t_init, u_init = self.get_train_dataset()
        
        restore_best_weights = Callback_EarlyStopping(self.dnn, self.coeff_net)
        
        callbacks = [restore_best_weights]
        
        for ep in tqdm(range(self.epoch)):
            if ep % self.resample_freq == 0:
                x_res, t_res, x_upper, t_upper, x_lower, t_lower, x_dat, t_dat, u_dat, x_init, t_init, u_init = self.get_train_dataset()
     
                pde_grad, ubc_grad, lbc_grad, data_grad, ic_grad, dnn_reg_grad, coeff_net_reg_grad = self.loss_grad( x_res, t_res, x_upper, t_upper, x_dat, t_dat, u_dat, x_lower, t_lower, x_init, t_init, u_init)
                
                self.update_weights(pde_grad, ubc_grad, lbc_grad, data_grad, ic_grad, dnn_reg_grad, coeff_net_reg_grad)


            total_loss, pde_loss, data_loss, ubc_loss, lbc_loss, ic_loss = self.train_step(x_res, t_res, x_upper, t_upper, x_lower, t_lower, x_dat, t_dat, u_dat, x_init, t_init, u_init)


            if ep % self.mntr == 0:
                elps = time.time() - t0
                self.ep_log.append(ep) 
                self.total_loss_log.append(total_loss.numpy())
                self.residual_log.append(pde_loss.numpy())
                self.data_loss_log.append(data_loss.numpy())
                self.ub_loss_log.append(ubc_loss.numpy())
                self.lb_loss_log.append(lbc_loss.numpy())
                self.init_loss_log.append(ic_loss.numpy())

                print("ep:%d, total_loss: %.4e, residual_loss: %.5e, data_loss %.5e, ubc_loss %.5e, lower_loss %.5e, init_loss: %.5e, elps: %.3f"%(ep, total_loss.numpy(), pde_loss.numpy(), data_loss.numpy(), ubc_loss.numpy(), lbc_loss.numpy(), ic_loss.numpy(), elps))
                
                t0 = time.time()
                
            losses = [pde_loss.numpy(), data_loss.numpy(), ic_loss.numpy(), lbc_loss.numpy(), ubc_loss.numpy()]
            # loss_weights = [self.adapt_data,self.adapt_ic,self.adapt_lbc,self.adapt_ubc, self.adapt_dnn_reg, self.adapt_coef_reg]

            for callback in callbacks:
                callback.on_epoch_end(ep,total_loss.numpy(),losses)        #callback.on_epoch_end(ep,total_loss.numpy(),losses,loss_weights)
            
            if restore_best_weights.should_stop:
                print(f"Stopping early after {ep} epochs.")
                break
        print(f"\n ***************Restoring to Best Weights at epoch: {restore_best_weights.best_epoch} with Total Loss : {restore_best_weights.best}**************")
        
        self.dnn.set_weights(restore_best_weights.best_weights_1)
        self.coeff_net.set_weights(restore_best_weights.best_weights_2)
        
        print("\n ***************TRAINING COMPLETED**************\n")
        


        
        loss_logs = [self.ep_log, self.total_loss_log, self.residual_log, self.data_loss_log, self.ub_loss_log, self.init_loss_log]
        #adaptive_weight_logs = [self.ep_log, self.weights_list_data, self.weights_list_ic, self.weights_list_lbc, self.weights_list_ubc, self.weight_list_dnn, self.weight_list_coef]
        best_epoch = restore_best_weights.best_epoch
        best_losses = restore_best_weights.best_losses
        best_logs = best_losses + [best_epoch]
        return loss_logs, best_logs
    
    def infer_in_batches(self):
        x_infer = np.linspace(self.x_min, self.x_max, 1000)
        t_infer = np.linspace(self.t_min, self.t_max, 1000)

        x, t = np.meshgrid(x_infer, t_infer)
        x_flat = x.reshape(-1, 1)
        t_flat = t.reshape(-1, 1)

        # Convert to tensors once
        x_flat = tf.convert_to_tensor(x_flat, dtype=tf.float32)
        t_flat = tf.convert_to_tensor(t_flat, dtype=tf.float32)

        # Prepare storage for chunked predictions
        u_hat_list = []
        d_hat_list = []

        batch_size = 20000
        n_total = x_flat.shape[0]  # 1,000,000 for a 1000x1000 grid

        # Loop over the entire dataset in chunks
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            
            # Slice a batch of x and t
            x_batch = x_flat[start:end]
            t_batch = t_flat[start:end]

            # 1) Predict solution for this batch
            u_hat_batch = self.predict(x_batch, t_batch)  # shape: [batch_size, ...]

            # 2) Predict coefficient from that solution
            d_hat_batch = self.predict_coeff(u_hat_batch)  # shape: [batch_size, ...]

            # Collect partial results
            u_hat_list.append(u_hat_batch)
            d_hat_list.append(d_hat_batch)

        # Concatenate the batch-wise results into one large tensor
        u_hat_full = tf.concat(u_hat_list, axis=0)  # shape: [n_total, ...]
        d_hat_full = tf.concat(d_hat_list, axis=0)  # shape: [n_total, ...]

        # Now reshape from (n_total,) to (1000,1000) 
        # (assuming each prediction is scalar per point)
        u_hat_array = u_hat_full.numpy().reshape(x.shape)
        d_hat_array = d_hat_full.numpy().reshape(x.shape)

        return u_hat_array, d_hat_array

        

