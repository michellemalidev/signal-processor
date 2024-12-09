from src.ekg_processing import EKGProcessor
from src.heart_rate_calculation import HeartRateCalculator
import numpy as np

def main():
    # Configuration
    sampling_rate = 250  # Hz
    file_path = 'data/ekg_signal.csv'

    # Initialize processors
    processor = EKGProcessor(sampling_rate)
    hr_calculator = HeartRateCalculator(sampling_rate)

    try:
        # Load and process signal
        raw_signal = processor.load_data(file_path)
        if raw_signal is None:
            print("Failed to load EKG data")
            return

        # Preprocess the signal
        processed_signal = processor.preprocess_signal(raw_signal)

        # Calculate heart rate
        heart_rate, peaks = hr_calculator.calculate_heart_rate(processed_signal)
        print(f"Calculated Heart Rate: {heart_rate:.2f} BPM")

        # Visualize results
        hr_calculator.plot_signal_with_peaks(processed_signal, peaks)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
