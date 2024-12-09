import numpy as np
from scipy.signal import butter, filtfilt

class EKGProcessor:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def load_data(self, file_path):
        """Load EKG data from a CSV file."""
        try:
            data = np.loadtxt(file_path, delimiter=',')
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def apply_bandpass_filter(self, signal, lowcut=0.5, highcut=50.0, order=2):
        """Apply bandpass filter to remove noise."""
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def normalize_signal(self, signal):
        """Normalize the signal to have zero mean and unit variance."""
        return (signal - np.mean(signal)) / np.std(signal)

    def preprocess_signal(self, signal):
        """Complete preprocessing pipeline."""
        # Apply bandpass filter
        filtered_signal = self.apply_bandpass_filter(signal)
        # Normalize the filtered signal
        normalized_signal = self.normalize_signal(filtered_signal)
        return normalized_signal
