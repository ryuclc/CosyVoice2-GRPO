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
# from tqdm import tqdm


def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    rank = int(os.environ['LOCAL_RANK'])
    threads_num = int(os.environ['WORLD_SIZE'])
    out_dir = os.path.join(args.dir, 'spk_embedding')
    logger.info("rank {}/{}: out_dir {}.".format(
        rank, threads_num, out_dir
    ))
    if out_dir is not None:
        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            while not os.path.exists(out_dir):
                time.sleep(0.5)
    
    utt2wav, spk2utt = {}, {}
    with open('{}/embedding.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/spk2utt'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            spk2utt[l[0]] = l[1:]

    all_recs = list(spk2utt.keys())
    local_all_recs = all_recs[rank::threads_num]

    out_path = os.path.join(out_dir, f"spk_embedding.{rank:02d}")
    wav_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_path}.ark,{out_path}.scp")
    out_path = os.path.join(out_dir, f"length.{rank:02d}.txt")
    length_writer = open(out_path, "wt")
    meeting_count = 0
    for i, spk in enumerate(local_all_recs):

        spk2embedding = []
        for utt in spk2utt[spk]:
            if utt not in utt2wav:
                continue
            spk2embedding.append(kaldiio.load_mat(utt2wav[utt]))
        
        if len(spk2embedding)==0:
            continue
        
        mean_embedding = np.array(spk2embedding).mean(axis=0)
        wav_writer(spk, mean_embedding.astype(np.float32))

        length_writer.write("{} {}\n".format(spk, len(spk2embedding)))

        if i % 100 == 0:
            logger.info("{}/{}: process {}.".format(rank, threads_num, spk))

        meeting_count += 1

    wav_writer.close()
    length_writer.close()
    logger.info("{}/{}: Complete {} records.".format(rank, threads_num, meeting_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    main(args)
