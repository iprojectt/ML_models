


import librosa
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


import librosa
import numpy as np
import os
import pandas as pd
from hmmlearn import hmm

# Path to the folder containing audio recordings
data_dir = "../../data/external/recordings"
# making directories to store MFCC features
test_mfcc_dir = "../../data/interim/testing_mfcc"
train_mfcc_dir = "../../data/interim/training_mfcc"
os.makedirs(test_mfcc_dir, exist_ok=True)
os.makedirs(train_mfcc_dir,exist_ok=True)

# Dictionary to store MFCCs for each file
mfcc_features = {}
# Set a smaller n_fft value
n_fft_value = 512  
# Iterate through each .wav file in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(data_dir, file_name)
        # Extract iteration number from the filename
        try:
            iteration_num = int(file_name.rsplit("_", 1)[1].split(".")[0])
        except (IndexError, ValueError):
            print(f"Skipping {file_name}: Unable to parse iteration number.")
            continue
        
        # Load the audio file
        y, sr = librosa.load(file_path, sr=8000)
        # Extract MFCC features with adjusted n_fft
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft_value)
        # Decide destination based on iteration number
        if iteration_num <= 4:
            # Testing samples
            dest_folder = test_mfcc_dir
        else:
            # Training samples
            dest_folder = train_mfcc_dir
        # Save MFCC feature as a CSV for each file
        dest_path = os.path.join(dest_folder, f"{file_name}.csv")
        pd.DataFrame(mfcc).to_csv(dest_path, index=False)
        # print(f"Saved MFCC for {file_name} to {dest_path}")
        






output_dir = "../../assignments/5/figures"

import random
# Define the number of samples per digit to visualize
samples_per_digit = 1
# Create a dictionary to store file paths for each digit (0-9)
digit_files = {str(i): [] for i in range(10)}
# Populate the dictionary with file paths for each digit
for file_name in os.listdir(data_dir):
    if file_name.endswith('.wav'):
        digit = file_name.split('_')[0]  # Assuming file names are in the format 'digit_*.wav'
        if digit in digit_files:
            digit_files[digit].append(file_name)
# Randomly select a few files per digit
selected_files = []
for digit, files in digit_files.items():
    selected_files.extend(random.sample(files, min(samples_per_digit, len(files))))
# Plot MFCCs for the selected files
for file_name in selected_files:
    file_path = os.path.join(data_dir, file_name)
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512)
    # Plotting the MFCCs as a heatmap for visualization
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"MFCC for {file_name}")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    # Save the plot in the specified directory
    save_path = os.path.join(output_dir, f"{file_name}_MFCC.png")
    plt.savefig(save_path)
    plt.show()




import os
import numpy as np
import pandas as pd
from hmmlearn import hmm


# Paths to the directories containing MFCC files for training and testing
train_mfcc_dir = "../../data/interim/training_mfcc"
test_mfcc_dir = "../../data/interim/testing_mfcc"
# Dictionary to store HMM models for each digit
hmm_models = {}
# Function to load MFCC data from CSV files
def load_mfcc_data(directory):
    data = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            # Extract digit label from the file name
            digit = int(file_name.split("_")[0])
            file_path = os.path.join(directory, file_name)
            mfcc_features = pd.read_csv(file_path).values
            # Append the MFCC features for each digit
            if digit not in data:
                data[digit] = []
            data[digit].append(mfcc_features)
    return data

# Load MFCC features from the training directory
train_data = load_mfcc_data(train_mfcc_dir)
# Find the maximum number of frames across all samples in the entire dataset
all_mfcc = [mfcc for mfcc_list in train_data.values() for mfcc in mfcc_list]
max_len = max(mfcc.shape[1] for mfcc in all_mfcc)
# Train an HMM model for each digit
for digit, mfcc_list in train_data.items():
    padded_mfcc_list = [np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant') for mfcc in mfcc_list]
    padded_mfcc_list = [mfcc.T for mfcc in padded_mfcc_list]  # Transpose each array
    X = np.vstack(padded_mfcc_list)  # Stack along the frame (row) dimension
    lengths = [mfcc.shape[0] for mfcc in padded_mfcc_list]  # Update lengths after padding
    # # Debugging: Print total rows in X and sum of lengths
    # print(f"Total number of samples in X: {X.shape}")
    # print(f"Sum of lengths: {sum(lengths)}")
    # Ensure the sum of lengths matches X's row count
    if X.shape[0] != sum(lengths):
        raise ValueError("The sum of lengths does not match the total number of rows in X.")
    # Create an HMM model (adjust n_components as needed based on dataset complexity)
    model = hmm.GaussianHMM(n_components=10, covariance_type='diag', n_iter=1000)
    model.fit(X, lengths)  # Train the HMM with the data for this digit
    hmm_models[digit] = model
    print(f"Trained HMM model for digit {digit}")

def predict_digit(mfcc_features, models):
    max_score = float('-inf')
    best_digit = None
    for digit, model in models.items():
        try:
            score = model.score(mfcc_features)
            # print(f"Digit: {digit}, Score: {score}")  # Debugging line
            if score > max_score:
                max_score = score
                best_digit = digit
        except Exception as e:
            print(f"Error with model for digit {digit}: {e}")
            continue
    return best_digit


# Determine the maximum number of frames across all training samples to use for padding test samples
# Assuming max_len is the maximum length used in training
max_len = 36  
# Function to pad MFCC data for consistency
def pad_mfcc(mfcc_list, max_len):
    padded_mfcc_list = [np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant') for mfcc in mfcc_list]
    return [mfcc.T for mfcc in padded_mfcc_list]  # Transpose each padded MFCC
# Load and pad MFCC features from the test directory
test_data = load_mfcc_data(test_mfcc_dir)
padded_test_data = {digit: pad_mfcc(mfcc_list, max_len) for digit, mfcc_list in test_data.items()}
# Print the shape of padded MFCC features for each digit
# for digit, padded_mfcc_list in padded_test_data.items():
#     print(f"Digit: {digit}")
#     for i, padded_mfcc in enumerate(padded_mfcc_list):
#         print(f"  Sample {i+1} padded shape: {padded_mfcc.shape}")
# Evaluate the models on padded test samples and calculate accuracy
correct_predictions = 0
total_predictions = 0

for digit, padded_mfcc_list in padded_test_data.items():
    for mfcc_features in padded_mfcc_list:
        predicted_digit = predict_digit(mfcc_features, hmm_models)
        print(f"True digit: {digit}, Predicted digit: {predicted_digit}")
        if predicted_digit == digit:
            correct_predictions += 1
        total_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")








import os
import numpy as np
import scipy.io.wavfile

def trim_silence(audio, noise_threshold=150):
    """
    Trims silence from the beginning and end of an audio array.
    :param audio: numpy array of audio data (1D or 2D for mono/stereo)
    :param noise_threshold: amplitude threshold to define silence
    :return: a trimmed numpy array
    """
    # Convert stereo to mono by averaging channels if necessary
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(audio.dtype)

    start = next((idx for idx, point in enumerate(audio) if abs(point) > noise_threshold), None)
    end = next((idx for idx, point in enumerate(audio[::-1]) if abs(point) > noise_threshold), None)
    if start is None or end is None:
        return audio  # Return original array if no silence is detected
    return audio[start:len(audio) - end]


def trim_silence_file(file_path, noise_threshold=150):
    """
    Reads a WAV file, trims silence from the beginning and end, and overwrites the original file.

    :param file_path: path to the WAV file
    :param noise_threshold: amplitude threshold to define silence
    :return: None
    """
    rate, audio = scipy.io.wavfile.read(file_path)
    trimmed_audio = trim_silence(audio, noise_threshold=noise_threshold)
    scipy.io.wavfile.write(file_path, rate, trimmed_audio)
    # print(f"Trimmed file saved: {file_path}")


def process_directory(directory_path, noise_threshold=150):
    """
    Processes all `.wav` files in a given directory by trimming silence from the beginning and end.

    :param directory_path: Path to the directory containing `.wav` files
    :param noise_threshold: Amplitude threshold to define silence
    :return: None
    """
    # Check if the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError("The provided path is not a directory.")

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            # print(f"Processing file: {file_path}")
            trim_silence_file(file_path, noise_threshold=noise_threshold)


# Usage example
process_directory("../../data/interim/personal_recording")









# Paths to the directories for MFCC files and recordings
personal_recordings_dir = "../../data/interim/personal_recording"
personal_mfcc_dir = "../../data/interim/personal_mfcc"
os.makedirs(personal_mfcc_dir, exist_ok=True)

# Function to extract and save MFCC features for new recordings
def extract_mfcc_for_personal_recordings():
    for file_name in os.listdir(personal_recordings_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(personal_recordings_dir, file_name)
            
            try:
                # Extract digit from the filename (assuming format "digit_iteration.wav")
                digit = int(file_name.split("_")[0])
            except (IndexError, ValueError):
                print(f"Skipping {file_name}: Unable to parse digit.")
                continue

            # Adjust the parameters when extracting MFCC features
            y, sr = librosa.load(file_path, sr=10000)  # Set a higher sampling rate if needed
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, n_mels=60)  # Set a lower n_mels if needed
            # Save the MFCC feature as a CSV for each file
            dest_path = os.path.join(personal_mfcc_dir, f"{file_name}.csv")
            pd.DataFrame(mfcc).to_csv(dest_path, index=False)
            # print(f"Saved MFCC for {file_name} to {dest_path}")

# Extract MFCC features for personal recordings
extract_mfcc_for_personal_recordings()
# Load MFCC data for testing generalization
def load_mfcc_data(directory):
    data = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            # Extract digit label from the file name
            digit = int(file_name.split("_")[0])
            file_path = os.path.join(directory, file_name)
            mfcc_features = pd.read_csv(file_path).values
            # Append the MFCC features for each digit
            if digit not in data:
                data[digit] = []
            data[digit].append(mfcc_features)
    return data

# Load and pad MFCC features for personal recordings
personal_data = load_mfcc_data(personal_mfcc_dir)
# Set the maximum frame length (from training)
max_len = 36  # Adjust if different
# Padding or truncating function to ensure each MFCC has 36 frames
def pad_or_truncate_mfcc(mfcc_list, max_len):
    processed_mfcc_list = []
    for mfcc in mfcc_list:
        # Truncate if more than max_len frames
        if mfcc.shape[1] > max_len:
            mfcc = mfcc[:, :max_len]
        # Pad with zeros if fewer than max_len frames
        elif mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        processed_mfcc_list.append(mfcc.T)  # Transpose each to match required shape
    return processed_mfcc_list
# Pad or truncate MFCC features for personal recordings
padded_personal_data = {digit: pad_or_truncate_mfcc(mfcc_list, max_len) for digit, mfcc_list in personal_data.items()}
# Function to predict digit based on MFCC features
def predict_digit(mfcc_features, models):
    max_score = float('-inf')
    best_digit = None
    for digit, model in models.items():
        try:
            score = model.score(mfcc_features)
            if score > max_score:
                max_score = score
                best_digit = digit
        except Exception as e:
            print(f"Error with model for digit {digit}: {e}")
            continue
    return best_digit

# Evaluate on personal recordings
correct_predictions = 0
total_predictions = 0

for digit, padded_mfcc_list in padded_personal_data.items():
    for mfcc_features in padded_mfcc_list:
        predicted_digit = predict_digit(mfcc_features, hmm_models)
        print(f"True digit: {digit}, Predicted digit: {predicted_digit}")
        if predicted_digit == digit:
            correct_predictions += 1
        total_predictions += 1

# Calculate accuracy for personal recordings
accuracy = correct_predictions / total_predictions * 100
print(f"Generalization Accuracy on Personal Recordings: {accuracy:.2f}%")


