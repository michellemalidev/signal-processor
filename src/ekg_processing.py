import numpy as np
from scipy.signal import butter, filtfilt
import os

class EKGProcessor:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def validate_data(self, data):
        """Validate the EKG data.
        
        Args:
            data (numpy.ndarray): EKG data to validate
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data is None:
            print("Validation error: Data is None")
            return False
        if not isinstance(data, np.ndarray):
            print("Validation error: Data is not a numpy array")
            return False
        if data.ndim != 1:
            print("Validation error: Data is not one-dimensional")
            return False
        if np.any(np.isnan(data)):
            print("Validation error: Data contains NaN values")
            return False
        if np.any(np.isinf(data)):
            print("Validation error: Data contains infinite values")
            return False
        return True

    def load_data(self, file_path):
        """Load EKG data from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing EKG data
            
        Returns:
            numpy.ndarray: Loaded EKG data if successful, None otherwise
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file is empty or contains invalid data
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            data = np.loadtxt(file_path, delimiter=',')
            
            if not self.validate_data(data):
                return None
            
            if data.size == 0:
                raise ValueError("File is empty")
                
            return data
            
        except FileNotFoundError as e:
            print(f"File error: {e}")
            return None
        except ValueError as e:
            print(f"Data error: Invalid or empty data in file - {e}")
            return None
        except Exception as e:
            print(f"Unexpected error while loading data: {e}")
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
