import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class HeartRateCalculator:
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate

    def detect_r_peaks(self, signal):
        """Detect R-peaks in the EKG signal."""
        # Find peaks with minimum height and distance
        peaks, _ = find_peaks(signal, 
                            height=0.25,  # Adjust threshold as needed
                            distance=self.sampling_rate * 0.5)  # Minimum 0.5s between peaks
        return peaks

    def calculate_heart_rate(self, signal):
        """Calculate heart rate from EKG signal."""
        peaks = self.detect_r_peaks(signal)
        
        if len(peaks) < 2:
            return 0
        
        # Calculate RR intervals and convert to seconds
        rr_intervals = np.diff(peaks) / self.sampling_rate
        # Calculate average heart rate in BPM
        heart_rate = 60 / np.mean(rr_intervals)
        return heart_rate, peaks

    def plot_signal_with_peaks(self, signal, peaks, title="EKG Signal with Detected R-Peaks"):
        """Visualize the EKG signal with detected R-peaks."""
        plt.figure(figsize=(15, 5))
        time = np.arange(len(signal)) / self.sampling_rate
        
        plt.plot(time, signal, label='EKG Signal')
        plt.plot(peaks/self.sampling_rate, signal[peaks], 'ro', label='R-Peaks')
        
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
