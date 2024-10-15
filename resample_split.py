from google.colab import drive
drive.mount('/content/drive')

import os
import glob
import pandas as pd
import librosa
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map
import shutil

def resample(audio, orig_fs, target_fs=16000):
    """Resamples the audio to the target sample rate."""
    out = []
    for c in range(audio.shape[0]):  # For each channel
        tmp = audio[c].detach().cpu().numpy()
        if target_fs != orig_fs:
            tmp = librosa.resample(tmp, orig_sr=orig_fs, target_sr=target_fs)
        out.append(torch.from_numpy(tmp))
    return torch.stack(out)

def resample_file(file_path, target_fs=16000):
    """Loads and resamples a single audio file."""
    audio, orig_fs = torchaudio.load(file_path)
    return resample(audio, orig_fs, target_fs)

def resample_and_save(args):
    """Resamples a file and saves it, preserving folder structure."""
    file_path, original_base_dir, resampled_base_dir, target_fs = args
    resampled_audio = resample_file(file_path, target_fs)
    
    # Compute the relative path from the original base directory
    relative_path = os.path.relpath(file_path, start=original_base_dir)
    output_file = os.path.join(resampled_base_dir, relative_path)
    
    os.makedirs(Path(output_file).parent, exist_ok=True)
    torchaudio.save(output_file, resampled_audio, target_fs)

def resample_all_files(df, original_base_dir, resampled_base_dir, target_fs=16000):
    """Resamples all files and saves them to the new directory."""
    file_paths = construct_file_paths(df, original_base_dir)
    print(f"Starting resampling for {len(file_paths)} files into {resampled_base_dir}...")
    
    # Prepare arguments for multiprocessing
    workers_args = [(f, original_base_dir, resampled_base_dir, target_fs) for f in file_paths]
    
    n_workers = min(10, mp.cpu_count())
    process_map(resample_and_save, workers_args, max_workers=n_workers, chunksize=1)

def generate_durations(audio_dir, out_csv):
    """Generates a CSV file with the duration of each audio file."""
    meta_list = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                d = sf.info(file_path).duration
                # Use relative path from audio_dir
                relative_file_path = os.path.relpath(file_path, audio_dir)
                meta_list.append([relative_file_path, d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    meta_df.to_csv(out_csv, index=False, float_format="%.2f")
    return meta_df

def construct_file_paths(df, audio_base_folder):
    """Constructs full file paths for each audio file based on metadata."""
    file_paths = []
    for _, row in df.iterrows():
        folder = f"fold{row['fold']}"
        file_name = row['slice_file_name']
        file_path = os.path.join(audio_base_folder, folder, file_name)
        file_paths.append(file_path)
    return file_paths

def copy_files(file_paths, source_base_dir, destination_folder):
    """Copies files to the destination folder, preserving the folder structure."""
    for file_path in file_paths:
        relative_path = os.path.relpath(file_path, start=source_base_dir)
        dest_path = os.path.join(destination_folder, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file_path, dest_path)

def split_data(df, resampled_base_dir, train_folder, val_folder, test_folder, train_csv, val_csv, test_csv):
    """Splits the dataset into train, validation, and test sets based on fold number."""
    # Filter data based on fold number
    train_data = df[df['fold'] <= 8]
    val_data = df[df['fold'] == 9]
    test_data = df[df['fold'] == 10]
    
    # Construct file paths for resampled files
    train_file_paths = construct_file_paths(train_data, resampled_base_dir)
    val_file_paths = construct_file_paths(val_data, resampled_base_dir)
    test_file_paths = construct_file_paths(test_data, resampled_base_dir)
    
    # Copy files to respective folders
    print("Copying training files...")
    copy_files(train_file_paths, resampled_base_dir, train_folder)
    print("Copying validation files...")
    copy_files(val_file_paths, resampled_base_dir, val_folder)
    print("Copying test files...")
    copy_files(test_file_paths, resampled_base_dir, test_folder)
    
    # Save CSVs for each split (keeping fold number and annotation information)
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    test_data.to_csv(test_csv, index=False)

def merge_metadata_and_durations(metadata_csv, durations_csv, output_csv):
    """Merges metadata and duration CSV files and saves the result."""
    metadata_df = pd.read_csv(metadata_csv)
    durations_df = pd.read_csv(durations_csv)
    # Extract the slice_file_name from the 'filename' in durations_df
    durations_df['slice_file_name'] = durations_df['filename'].apply(os.path.basename)
    # Merge on 'slice_file_name'
    merged_df = pd.merge(metadata_df, durations_df[['slice_file_name', 'duration']], on='slice_file_name')
    merged_df.to_csv(output_csv, index=False)

def process_data(config_data):
    """Main function to process the data."""
    # Load metadata
    urban_sound_df = pd.read_csv(config_data['audio_csv'])

    # Resample all files
    resample_all_files(
        urban_sound_df,
        config_data['audio_folder_8k'],
        config_data['resampled_base_dir'],
        config_data['fs']
    )

    # Split data into train, validation, and test sets
    split_data(
        urban_sound_df,
        config_data['resampled_base_dir'],
        config_data['audio_train_folder'],
        config_data['audio_val_folder'],
        config_data['test_folder'],
        config_data['audio_train_csv'],
        config_data['audio_val_csv'],
        config_data['test_csv']
    )

    # Generate duration CSV files for each split
    generate_durations(config_data['audio_train_folder'], config_data['audio_train_dur'])
    generate_durations(config_data['audio_val_folder'], config_data['audio_val_dur'])
    generate_durations(config_data['test_folder'], config_data['test_dur'])

    # Merge metadata and durations for each split
    merge_metadata_and_durations(config_data['audio_train_csv'], config_data['audio_train_dur'], config_data['train_full_csv'])
    merge_metadata_and_durations(config_data['audio_val_csv'], config_data['audio_val_dur'], config_data['val_full_csv'])
    merge_metadata_and_durations(config_data['test_csv'], config_data['test_dur'], config_data['test_full_csv'])

if __name__ == "__main__":
    # Configuration dictionary
    config_data = {
        # Original audio and metadata files
        "audio_folder_8k": "/content/drive/MyDrive/UrbanSound8K/audio",
        "audio_csv": "/content/drive/MyDrive/UrbanSound8K/metadata/UrbanSound8K.csv",

        # Resampled data directory (no 'audio' folder inside)
        "resampled_base_dir": "/content/drive/MyDrive/UrbanSound8K/audio_16k",

        # Paths for resampled and split data (no extra 'audio' folder)
        "audio_train_folder": "/content/drive/MyDrive/UrbanSound8K/train/audio_16k",
        "audio_train_csv": "/content/drive/MyDrive/UrbanSound8K/train/train.csv",
        "audio_val_folder": "/content/drive/MyDrive/UrbanSound8K/validation/audio_16k",
        "audio_val_csv": "/content/drive/MyDrive/UrbanSound8K/validation/validation.csv",
        "test_folder": "/content/drive/MyDrive/UrbanSound8K/test/audio_16k",
        "test_csv": "/content/drive/MyDrive/UrbanSound8K/test/test.csv",

        # Additional settings
        "fs": 16000,  # Target sampling rate

        # Output durations CSV files
        "audio_train_dur": "/content/drive/MyDrive/UrbanSound8K/train/train_durations.csv",
        "audio_val_dur": "/content/drive/MyDrive/UrbanSound8K/validation/val_durations.csv",
        "test_dur": "/content/drive/MyDrive/UrbanSound8K/test/test_durations.csv",

        # Output merged CSV files (metadata + durations)
        "train_full_csv": "/content/drive/MyDrive/UrbanSound8K/train/train_full.csv",
        "val_full_csv": "/content/drive/MyDrive/UrbanSound8K/validation/val_full.csv",
        "test_full_csv": "/content/drive/MyDrive/UrbanSound8K/test/test_full.csv",
    }

    process_data(config_data)
