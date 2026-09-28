import glob
import os
import argparse
import json
import gc
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.pcs400 import cal_pcs
import soundfile as sf

from utils.streaming import StreamingSTFT, StreamingISTFT
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    max_chunk_samples = int(2 * sampling_rate)

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ): 
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            # No utterance-level normalization, matching the training setup: it would
            # require the full utterance and break causal streaming.
            norm_factor = torch.tensor(1.0)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)

            if args.streaming:
                if args.chunk_size > max_chunk_samples:
                    print(f"WARNING: chunk_size ({args.chunk_size}) > 2 seconds. "
                          f"Capping to {max_chunk_samples} samples.")
                    args.chunk_size = max_chunk_samples
                # Validate chunk size compatibility
                if args.chunk_size < win_size:
                    print(f"WARNING: chunk_size ({args.chunk_size}) is smaller than win_size ({win_size}). "
                          f"This may cause issues with STFT processing. Recommended: chunk_size >= {win_size}")
                
                # Stateful streaming inference with InferenceParams
                stft_stream = StreamingSTFT(n_fft, hop_size, win_size, compress_factor, device)
                istft_stream = StreamingISTFT(n_fft, hop_size, win_size, compress_factor, device)
                num_samples = noisy_wav.size(-1)
                
                if args.streaming_mode == "realtime":
                    model.set_streaming(True, cache_seconds=args.transformer_cache_seconds)
                    model.reset_streaming_state()

                    for start in range(0, num_samples, args.chunk_size):
                        chunk = noisy_wav[:, start : start + args.chunk_size]
                        stft_out = stft_stream.process(chunk.squeeze(0))
                        if stft_out is None:
                            continue
                        noisy_mag_chunk, noisy_pha_chunk, _ = stft_out

                        # Frame-by-frame for true streaming with cached states.
                        for t in range(noisy_mag_chunk.shape[-1]):
                            mag_frame = noisy_mag_chunk[:, :, t : t + 1]
                            pha_frame = noisy_pha_chunk[:, :, t : t + 1]
                            amp_g, pha_g, _ = model(mag_frame, pha_frame, inference_params=None)
                            istft_stream.add_frames(amp_g, pha_g)
                else:
                    # Chunked processing (lower overhead, not true frame-by-frame).
                    for start in range(0, num_samples, args.chunk_size):
                        chunk = noisy_wav[:, start : start + args.chunk_size]
                        stft_out = stft_stream.process(chunk.squeeze(0))
                        if stft_out is None:
                            continue
                        noisy_mag_chunk, noisy_pha_chunk, _ = stft_out

                        # Process chunk (explicitly pass None for inference_params)
                        amp_g, pha_g, _ = model(noisy_mag_chunk, noisy_pha_chunk, inference_params=None)
                        istft_stream.add_frames(amp_g, pha_g)

                flush_out = stft_stream.flush()
                if flush_out is not None:
                    noisy_mag_chunk, noisy_pha_chunk, _ = flush_out
                    if args.streaming_mode == "realtime":
                        for t in range(noisy_mag_chunk.shape[-1]):
                            mag_frame = noisy_mag_chunk[:, :, t : t + 1]
                            pha_frame = noisy_pha_chunk[:, :, t : t + 1]
                            amp_g, pha_g, _ = model(mag_frame, pha_frame, inference_params=None)
                            istft_stream.add_frames(amp_g, pha_g)
                    else:
                        amp_g, pha_g, _ = model(noisy_mag_chunk, noisy_pha_chunk, inference_params=None)
                        istft_stream.add_frames(amp_g, pha_g)

                audio_g = istft_stream.finalize(length=num_samples).unsqueeze(0)
            else:
                noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
                amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
                audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)

            if args.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
            else:
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')

            del noisy_wav, audio_g, norm_factor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='recipes/KD1/KD1.yaml')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--post_processing_PCS', type=str2bool, default=False)
    parser.add_argument('--streaming', type=str2bool, default=False)
    parser.add_argument('--chunk_size', type=int, default=1600)  # ~100ms at 16kHz
    parser.add_argument('--streaming_mode', choices=['realtime', 'chunked'], default='realtime')
    parser.add_argument('--transformer_cache_seconds', type=float, default=2.0)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        

    inference(args, device)


if __name__ == '__main__':
    main()
