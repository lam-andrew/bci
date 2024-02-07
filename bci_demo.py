
""" 0. LOAD PRE-RECORDED DATA """
import time
import numpy as np
# Example for loading data from a CSV file
eeg_data = np.loadtxt('path/to/your/eeg_data.csv', delimiter=',')

# Example for loading data from a NPY file
eeg_data = np.load('path/to/your/eeg_data.npy')

# Example for loading data from an EDF file using MNE
import mne
raw = mne.io.read_raw_edf('path/to/your/eeg_data.edf')
eeg_data = raw.get_data()


""" 1. SIMULATE REAL-TIME DATA STREAM """
# Define a generator function to simulate real-time data chunks
def data_stream(eeg_data, fs, chunk_size=256):
    """Yield chunks of EEG data."""
    n_samples = eeg_data.shape[1]
    for start in range(0, n_samples, chunk_size):
        end = start + chunk_size
        yield eeg_data[:, start:end]
        # Simulate delay
        time.sleep(chunk_size / fs)
