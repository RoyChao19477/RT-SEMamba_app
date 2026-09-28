import os
import json
import argparse

def list_files_in_directory(directory_path):
    # List all files in the directory
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def save_files_to_json(files, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(files, json_file, indent=4)

def make_json(directory_path, output_file):
    # Get the list of files and save to JSON
    files = list_files_in_directory(directory_path)
    if not files:
        print(f"WARNING: no .wav files found under {directory_path} -> {output_file} will be empty")
    else:
        print(f"{output_file}: {len(files)} files from {directory_path}")
    save_files_to_json(files, output_file)

# create training set json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prefix_path',
        required=True,
        help="Root of the 16 kHz VCTK-DEMAND corpus. Must contain "
             "clean_trainset_28spk_wav_16k/, noisy_trainset_28spk_wav_16k/, "
             "clean_testset_wav_16k/, and noisy_testset_wav_16k/."
    )

    args = parser.parse_args()

    prefix = args.prefix_path
    if not os.path.isdir(prefix):
        raise SystemExit(f"--prefix_path is not a directory: {prefix}")


    ## You can manualy modify the clean and noisy path
    # ----------------------------------------------------------#
    ## train_clean
    make_json(
        os.path.join(prefix, 'clean_trainset_28spk_wav_16k/'),
        'data/train_clean.json'
    )

    ## train_noisy
    make_json(
        os.path.join(prefix, 'noisy_trainset_28spk_wav_16k/'),
        'data/train_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create valid set json
    ## valid_clean
    make_json(
         os.path.join(prefix, 'clean_testset_wav_16k/'),
        'data/valid_clean.json'
    )

    ## valid_noisy
    make_json(
        os.path.join(prefix, 'noisy_testset_wav_16k/'),
        'data/valid_noisy.json'
    )
    # ----------------------------------------------------------#

    # ----------------------------------------------------------#
    # create testing set json
    ## test_clean
    make_json(
       os.path.join(prefix, 'clean_testset_wav_16k/'),
        'data/test_clean.json'
    )

    ## test_noisy
    make_json(
       os.path.join(prefix, 'noisy_testset_wav_16k/'),
        'data/test_noisy.json'
    )
    # ----------------------------------------------------------#


if __name__ == '__main__':
    main()
