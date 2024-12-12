import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


class Graphs:
    def plot_spectrogram(self, file_path, file_type):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Create the spectrogram
        spectrogram = librosa.stft(y)  # Short-time Fourier transform
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)  # Convert to decibel scale

        # Plot the spectrogram
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Spectrogram ({file_type.upper()})")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    # # Example usage for an MP3 file
    # plot_spectrogram("Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000000.mp3", "mp3")
    #
    # # Example usage for a WAV file
    # plot_spectrogram("000000.wav", "wav")

    # Function to plot side-by-side spectrograms for comparison
    def overlay_spectrograms(self, mp3_path, wav_path):
        # Load MP3 file
        y_mp3, sr_mp3 = librosa.load(mp3_path, sr=None)
        spectrogram_mp3 = librosa.stft(y_mp3)
        spectrogram_db_mp3 = librosa.amplitude_to_db(np.abs(spectrogram_mp3), ref=np.max)

        # Load WAV file
        y_wav, sr_wav = librosa.load(wav_path, sr=None)
        spectrogram_wav = librosa.stft(y_wav)
        spectrogram_db_wav = librosa.amplitude_to_db(np.abs(spectrogram_wav), ref=np.max)

        # Ensure both spectrograms are aligned by time and frequency
        min_time = min(spectrogram_db_mp3.shape[1], spectrogram_db_wav.shape[1])
        spectrogram_db_mp3 = spectrogram_db_mp3[:, :min_time]
        spectrogram_db_wav = spectrogram_db_wav[:, :min_time]

        # Plot the spectrograms
        plt.figure(figsize=(12, 8))

        # MP3 spectrogram in blue
        librosa.display.specshow(spectrogram_db_mp3, sr=sr_mp3, x_axis='time', y_axis='log', cmap='Blues', alpha=0.7)
        plt.colorbar(format="%+2.0f dB", label="MP3 Intensity (Blue)")

        # Overlay WAV spectrogram in yellow
        librosa.display.specshow(spectrogram_db_wav, sr=sr_wav, x_axis='time', y_axis='log', cmap='YlOrBr', alpha=0.5)
        plt.colorbar(format="%+2.0f dB", label="WAV Intensity (Yellow)")

        plt.title("Overlayed Spectrograms: MP3 (Blue) vs WAV (Yellow)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Provide paths to your MP3 and WAV files
    #overlay_spectrograms("Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000002.mp3", "000002.wav")


    def difference_spectrogram(self, mp3_path, wav_path):
        # Load MP3 file
        y_mp3, sr_mp3 = librosa.load(mp3_path, sr=None)
        spectrogram_mp3 = librosa.stft(y_mp3)
        spectrogram_db_mp3 = librosa.amplitude_to_db(np.abs(spectrogram_mp3), ref=np.max)

        # Load WAV file
        y_wav, sr_wav = librosa.load(wav_path, sr=None)
        spectrogram_wav = librosa.stft(y_wav)
        spectrogram_db_wav = librosa.amplitude_to_db(np.abs(spectrogram_wav), ref=np.max)

        # Ensure both spectrograms are aligned by time and frequency
        min_time = min(spectrogram_db_mp3.shape[1], spectrogram_db_wav.shape[1])
        spectrogram_db_mp3 = spectrogram_db_mp3[:, :min_time]
        spectrogram_db_wav = spectrogram_db_wav[:, :min_time]

        # Calculate absolute differences
        difference = np.abs(spectrogram_db_mp3 - spectrogram_db_wav)

        # Plot the differences
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(difference, sr=sr_mp3, x_axis='time', y_axis='log', cmap='coolwarm')
        plt.colorbar(format="%+2.0f dB", label="Difference Intensity")
        plt.title("Difference Spectrogram Between MP3 and WAV")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


    # Provide paths to your MP3 and WAV files
    #difference_spectrogram("Dataset/mozillaorg/common-voice/versions/2/cv-valid-test/cv-valid-test/sample-000002.mp3", "000002.wav")
