# This script dumps waveforms to wav-copy format wav ark, including sample rate and int16 sequence.
"""
Author: Zhihao Du
Date: 2023.03.29
Description: Dump wav, flac and ark files to wav ark files with given sampling rate.
"""
import logging
import warnings
warnings.filterwarnings("ignore")
import os
import time
import argparse
import numpy as np
import kaldiio
import torch
import torchaudio
torchaudio.set_audio_backend('soundfile')
import torchaudio.compliance.kaldi as kaldi
# from tqdm import tqdm
import onnxruntime


def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    rank = int(os.environ['LOCAL_RANK'])
    gpu_id = rank % torch.cuda.device_count() if torch.cuda.device_count()>0 else 0
    threads_num = int(os.environ['WORLD_SIZE'])
    sr, sample_bits = args.sample_rate, 16
    out_dir = args.out_dir
    logger.info("rank {}/{}: gpu_id {}, Sample rate {}, sample bits {}, out_dir {}.".format(
        rank, threads_num, gpu_id, sr, sample_bits, out_dir
    ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)
    
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
    
    all_recs = []
    if args.wav_scp is not None and len(args.wav_scp) > 0:
        for one_line in open(args.wav_scp, "rt", encoding="utf-8"):
            path = one_line.strip()
            key, path = path.split(" ", maxsplit=1)
            all_recs.append((key, path))
    else:
        for one_line in open(args.wav_list, "rt", encoding="utf-8"):
            path = one_line.strip()
            key = os.path.basename(path).rsplit(".", 1)[0]
            all_recs.append((key, path))
    all_recs.sort(key=lambda x: x[0])
    local_all_recs = all_recs[rank::threads_num]

    out_path = os.path.join(out_dir, f"embedding.{rank:02d}")
    wav_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_path}.ark,{out_path}.scp")
    # out_path = os.path.join(out_dir, f"length.{rank:02d}.txt")
    # length_writer = open(out_path, "wt")
    meeting_count = 0
    for i, (uttid, wav_path) in enumerate(local_all_recs):
        sr, sample_bits = args.sample_rate, 16
        # skip files not ending with wav
        # if not wav_path.lower().endswith(".wav") and \
        #         not wav_path.lower().endswith(".flac") and \
        #         not (".ark" in wav_path.lower() and ":" in wav_path):
        #     logger.warning("rank {}/{}: Skip {} since {} file format is not wav or flac or ark.".format(
        #         rank, threads_num, uttid, wav_path
        #     ))
        #     continue
        if not (".ark" in wav_path.lower() and ":" in wav_path):
            # Use librosa to deal with multi-channels and different sampling rate
            audio, sample_rate = torchaudio.load(wav_path)
            if sample_rate != 16000:
                audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
            feat = kaldi.fbank(audio,
                               num_mel_bins=80,
                               dither=0,
                               sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embedding = ort_session.run(None, {ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten()
        else:
            wav = kaldiio.load_mat(wav_path)
            if isinstance(wav, tuple):
                if isinstance(wav[0], int):
                    sr, wav = wav
                else:
                    wav, sr = wav
            if (np.max(np.abs(wav)) > 1.0) or args.force_rescale:
                wav = wav / np.max(np.abs(wav)) * 0.9
        
        wav_writer(uttid, embedding.astype(np.float32))
        # length_writer.write("{} {}\n".format(uttid, len(feats)))

        if i % 100 == 0:
            logger.info("{}/{}: process {}.".format(rank, threads_num, uttid))

        meeting_count += 1

    wav_writer.close()
    logger.info("{}/{}: Complete {} records.".format(rank, threads_num, meeting_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_list',
                        type=str,
                        default=None,
                        help="wav path list")
    parser.add_argument('--wav_scp',
                        type=str,
                        default=None,
                        help="kaldi-style wav path script")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="local rank of gpu")
    parser.add_argument('--out_dir',
                        type=str,
                        default=None,
                        help="The output dir to save rttms and wavs")
    parser.add_argument('--sample_rate',
                        type=int,
                        default=16_000,
                        help="The expected sample rate.")
    parser.add_argument("--output_suffix",
                        type=str,
                        default="",
                        help="The suffix added to the end of file name.")
    parser.add_argument("--force_rescale",
                        type=bool,
                        default=False,
                        help="force rescale")
    parser.add_argument("--onnx_path",
                        type=str,
                        default="",
                        help="onnx_path")
    args = parser.parse_args()
    main(args)
