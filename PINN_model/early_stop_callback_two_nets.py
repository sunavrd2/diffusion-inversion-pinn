
import tensorflow as tf

class Callback_EarlyStopping():
    def __init__(self, my_model_1, my_model_2, patience=2000, verbose=1, restore_best_weights=True):

        ''' args:
            my_model_1 = tf.keras.sequential()
            my_model_2 = tf.keras.sequential()'''
        
        self.my_model_1 = my_model_1
        self.my_model_2 = my_model_2 
        self.patience = patience
        self.best_weights_1 = None
        self.best_weights_2 = None
        self.best_epoch = 0
        self.wait = 0
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.best = float('inf')
        self.should_stop = False
        self.best_losses = []

    def on_epoch_end(self, epoch, total_loss=None, losses = None):
        current = total_loss
        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights_1 = self.my_model_1.get_weights()
                self.best_weights_2 = self.my_model_2.get_weights()
                self.best_losses = losses
        else:
            self.wait += 1
            if self.wait >= self.patience:
                
                self.should_stop = True
                if self.verbose:
                    print(f"Restoring model weights from the end of the best epoch: {self.best_epoch}.")
                    print(f"\nBest total loss : {self.best}" )
                    print(f"\nBest losses [r,d,i,l,u] are: {self.best_losses}" )
                if self.restore_best_weights:
                    self.my_model_1.set_weights(self.best_weights_1)
                    self.my_model_2.set_weights(self.best_weights_2)


