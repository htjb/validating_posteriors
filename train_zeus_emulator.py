from globalemu.preprocess import process
from globalemu.network import nn
import numpy as np
from globalemu.eval import evaluate
from globalemu.plotter import signal_plot
import os


def gen_emulator(base_dir, AFB=True, resampling=True, resume=False):
        z = np.arange(10, 35, 0.1)
        if resume != True:
                process('full', z, base_dir=base_dir, 
                        data_location='zeus21_train_test/', logs=[1, 2, 5],
                        AFB=AFB, resampling=resampling)

        nn(batch_size=451, epochs=500, base_dir=base_dir, resume=resume, 
                layer_sizes=[32, 32, 32], input_shape=9, early_stop=True)

        predictor = evaluate(base_dir=base_dir, logs=[1, 2, 5])

        parameters = np.loadtxt('zeus21_train_test/test_data.txt')
        labels = np.loadtxt('zeus21_train_test/test_labels.txt')

        plotter = signal_plot(parameters, labels, 'rmse', predictor, base_dir,
                loss_label='RMSE = {:.4f} [mK]')

uber_dir = 'zeus_emulators/'
if not os.path.exists(uber_dir):
    os.makedirs(uber_dir)
gen_emulator(uber_dir + 'with_AFB_resampling/', AFB=True, resampling=True)

