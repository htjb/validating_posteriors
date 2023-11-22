from globalemu.preprocess import process
from globalemu.network import nn
import numpy as np
from globalemu.eval import evaluate
from globalemu.plotter import signal_plot


def gen_emulator(base_dir, AFB=True, resampling=True, resume=False):
        z = np.arange(6, 55, 0.1)
        if resume != True:
                process('full', z, base_dir=base_dir, 
                        data_location='signal_data/', logs=[0, 2, 4, 5],
                        AFB=AFB, resampling=resampling)

        nn(batch_size=451, epochs=500, base_dir=base_dir, resume=resume, 
                layer_sizes=[32, 32, 32], input_shape=9, early_stop=True)

        predictor = evaluate(base_dir=base_dir, logs=[0, 2, 4, 5])

        parameters = np.loadtxt('signal_data/test_data.txt')
        labels = np.loadtxt('signal_data/test_labels.txt')

        plotter = signal_plot(parameters, labels, 'rmse', predictor, base_dir,
                loss_label='RMSE = {:.4f} [mK]')

uber_dir = 'emulators/'
#gen_emulator(uber_dir + 'with_AFB_resampling/', AFB=True, resampling=True, resume=True)
#gen_emulator(uber_dir + 'with_AFB_only/', AFB=True, resampling=False)
#gen_emulator(uber_dir + 'with_resampling_only/', AFB=False, resampling=True, resume=True)
gen_emulator(uber_dir + 'no_AFB_no_resampling/', AFB=False, resampling=False)

