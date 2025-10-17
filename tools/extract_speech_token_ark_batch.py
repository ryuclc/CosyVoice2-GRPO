# This script dumps waveforms to wav-copy format wav ark, including sample rate and int16 sequence.
"""
Author: Zhihao Du
Date: 2023.03.29
Description: Dump wav, flac and ark files to wav ark files with given sampling rate.
"""
import logging
logging.basicConfig(level=logging.INFO)
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
# import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
# import torchaudio.compliance.kaldi as kaldi
# from tqdm import tqdm
# import onnxruntime
# import whisper
import s3tokenizer



class AudioDataset(Dataset):

    def __init__(self, wav_scp):
        self.data = []
        self.keys = []

        with open(wav_scp, 'r', encoding='utf-8') as f:
            for line in f:
                key, file_path = line.strip().split()
                self.data.append(file_path)
                self.keys.append(key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        key = self.keys[idx]
        audio = s3tokenizer.load_audio(file_path)
        mel = s3tokenizer.log_mel_spectrogram(audio)
        if len(audio) / 16000 > 30:
            print('do not support extract speech token for audio longer than 30s')
            is_too_long = True
            mel = mel[:,:3000]
        else:
            is_too_long = False
        return key, mel, is_too_long


def collate_fn(batch):
    keys = [item[0] for item in batch]
    mels = [item[1] for item in batch]
    is_too_longs = [item[2] for item in batch]
    mels, mels_lens = s3tokenizer.padding(mels)
    return keys, mels, mels_lens, is_too_longs


def init_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    print('Inference on multiple gpus, this gpu {}'.format(local_rank) +
          ', rank {}, world_size {}'.format(rank, world_size))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank



def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)
    rank = int(os.environ['LOCAL_RANK'])
    # gpu_id = rank % torch.cuda.device_count() if torch.cuda.device_count()>0 else 0
    # threads_num = int(os.environ['WORLD_SIZE'])
    # sr, sample_bits = args.sample_rate, 16
    out_dir = args.out_dir
    # logger.info("rank {}/{}: gpu_id {}, Sample rate {}, sample bits {}, out_dir {}.".format(
    #     rank, threads_num, gpu_id, sr, sample_bits, out_dir
    # ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)
    
    assert (torch.cuda.is_available())
    world_size, local_rank, rank = init_distributed()
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda")
    model = s3tokenizer.load_model(args.onnx_path).to(device)
    dataset = AudioDataset(args.wav_scp)

    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    sampler = DistributedSampler(dataset,
                                     num_replicas=world_size,
                                     rank=rank)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            sampler=sampler,
                            shuffle=False,
                            num_workers=2,
                            prefetch_factor=4,
                            collate_fn=collate_fn)

    out_path = os.path.join(out_dir, f"speech_token.{rank:02d}")
    wav_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_path}.ark,{out_path}.scp")
    out_path = os.path.join(out_dir, f"length.{rank:02d}.txt")
    length_writer = open(out_path, "wt")

    total_steps = len(dataset)
    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")
    
    for keys, mels, mels_lens, is_too_longs in dataloader:
        codes, codes_lens = model(mels.to(device), mels_lens.to(device))
        for i, k in enumerate(keys):
            if is_too_longs[i]:
                print('remove speech token for audio longer than 30s')
                speech_token = np.array([])
            else:
                speech_token = codes[i, :codes_lens[i].item()].cpu().numpy()

            wav_writer(k, speech_token.astype(np.int32))
            length_writer.write("{} {}\n".format(k, len(speech_token)))

        if rank == 0:
            progress_bar.update(world_size * len(keys))

    if rank == 0:
        progress_bar.close()

    

    wav_writer.close()
    length_writer.close()
    dist.barrier()
    dist.destroy_process_group()


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
