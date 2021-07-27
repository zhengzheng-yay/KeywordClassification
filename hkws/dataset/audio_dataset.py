import os
import random
import numpy as np
import math
import soundfile as sf
import time
import yaml

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchaudio
import torchaudio.compliance.kaldi as kaldi
import datetime

def _time_mask(y, max_t, max_frames):
    start = random.randint(0, max_frames - 1)
    length = random.randint(1, max_t)
    end = min(max_frames, start + length)
    y[start:end, :] = 0
    return y

def _freq_mask(y, max_f, max_freq):
    start = random.randint(0, max_freq - 1)
    length = random.randint(1, max_f)
    end = min(max_freq, start + length)
    y[:, start:end] = 0
    return y

def _time_wrap(y, max_w, max_frames, max_freq):
    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize((max_freq,
                                                   max_frames - warped),
                                                   BICUBIC)
        y = np.concatenate((left, right), 0)
    return y

def _spec_augmentation(x,
                       warp_for_time=False,
                       num=0,
                       max_t=50,
                       max_f=10,
                       max_w=80):
    """ Deep copy x and do spec augmentation then return it
    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp
    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]
    spec_types = [0, 1, 2, 3]
    # 0: only freq mask 
    # 1: only time mask 
    # 2: both freq and time mask
    # 3: freq, time mask and time warp 
    for i in range(num):
        spec_type = random.choice(spec_types)
        if 0 == spec_type:
            y = _freq_mask(y, max_f, max_freq)
        elif 1 == spec_type:
            y = _time_mask(y, max_t, max_frames)
        elif 2 == spec_type:
            y = _time_mask(_freq_mask(y, max_f, max_freq), max_t, max_frames)
        elif 3 == spec_type:
            y = _time_mask(_freq_mask(y, max_f, max_freq), max_t, max_frames)
            if warp_for_time and max_frames > max_w * 2:
                y = _time_wrap(y, max_w, max_frames, max_freq)
    return y

def _spec_substitute(x, max_t=20, num_t_sub=3):
    """ Deep copy x and do spec substitute then return it
    Args:
        x: input feature, T * F 2D
        max_t: max width of time substitute
        num_t_sub: number of time substitute to apply
    Returns:
        augmented feature
    """
    y = np.copy(x)
    max_frames = y.shape[0]
    for i in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = y[start - pos:end - pos, :]
    return y

# add speed perturb when loading wav
# return augmented, sr
from multiprocessing import Manager
m=Manager()
FEATS_CACHE_DICT=m.dict()
def _load_wav_with_speed(wav_file, speed, wav_id=None, cache=True):
    """ Load the wave from file and apply speed perpturbation
    Args:
        wav_file: input feature, T * F 2D
    Returns:
        augmented feature
    """
    
    if speed == 1.0:
        return torchaudio.load_wav(wav_file)
    else:
        si, _ = torchaudio.info(wav_file)
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.append_effect_to_chain('speed', speed)
        E.append_effect_to_chain("rate", si.rate)
        E.set_input_file(wav_file)
        wav, sr = E.sox_build_flow_effects()
        # sox will normalize the waveform, scale to [-32768, 32767]
        wav = wav * (1 << 15)
        return wav, sr

def _extract_feature(filename, speed_perturb, feature_extraction_conf, cache=True, fileid=""):
    speed = 1.0
    if speed_perturb:
        speeds = [1.0, 1.1, 0.9]
        weights = [1, 1, 1]
        speed = random.choices(speeds, weights, k=1)[0]
    feat_id = fileid+str(speed)
    if cache:
        if feat_id in FEATS_CACHE_DICT:
            return FEATS_CACHE_DICT[feat_id]

    waveform, sample_rate = _load_wav_with_speed(filename, speed)
    feature_type = feature_extraction_conf['feature_type']
    if "mfcc" == feature_type:
        mat = kaldi.mfcc(
            waveform,
            num_ceps=feature_extraction_conf['num_ceps'],
            num_mel_bins=feature_extraction_conf['num_mel_bins'],
            frame_length=feature_extraction_conf['frame_length'],
            frame_shift=feature_extraction_conf['frame_shift'],
            dither=feature_extraction_conf['wav_dither'],
            energy_floor=feature_extraction_conf['energy_floor'],
            sample_frequency=sample_rate
        )
    elif "fbank" == feature_type:
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=feature_extraction_conf['num_mel_bins'],
            frame_length=feature_extraction_conf['frame_length'],
            frame_shift=feature_extraction_conf['frame_shift'],
            dither=feature_extraction_conf['wav_dither'],
            energy_floor=feature_extraction_conf['energy_floor'],
            sample_frequency=sample_rate
        )
    else:
        print("We do not support this kind of feature type:%s\n"%(feature_type))
    mat = mat.detach().numpy()
    if cache:
        FEATS_CACHE_DICT[feat_id] = mat
    return mat

class WavDataset(Dataset):
    def __init__(self, 
                torch_scp_file, 
                with_label=True, 
                shuffle=True,
                feature_extraction_conf=None,
                feature_dither=0.0,
                speed_perturb=False,
                spec_aug=False,
                spec_aug_conf=None,
                spec_sub=False,
                spec_sub_conf=None
                ):
        self.items = [item.strip() for item in open(torch_scp_file)] # [(utt_id, wav_path)]
        self.dataset_size = len(self.items)
        self.shuffle = shuffle
        self.with_label = with_label
        
        self.feature_extraction_conf = feature_extraction_conf
        self.feature_dither = feature_dither
        self.speed_perturb = speed_perturb
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.spec_sub = spec_sub
        self.spec_sub_conf = spec_sub_conf
        
        if shuffle:
            random.shuffle(self.items)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        assert 0 <= idx and idx < self.dataset_size, "invalid index"       
        item = self.items[idx]
        utt = item.split()[0]
        filename = item.split()[1]

        feat = _extract_feature(filename, self.speed_perturb, self.feature_extraction_conf, cache=True, fileid=utt)
        # optional feature dither d ~ (-a, a) on fbank feature
        # a ~ (0, 0.5)
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            feat = feat + (np.random.random_sample(feat.shape) - 0.5) * a 

        # optinoal: spec substitute
        if self.spec_sub:
            feat = _spec_substitute(feat, **self.spec_sub_conf)

        # optinoal: spec augmentation
        if self.spec_aug:
            feat = _spec_augmentation(feat, **self.spec_aug_conf)

        if self.with_label:
            label= item.split()[2]
            return utt, feat, int(label)
        else:
            return utt, feat

def collate_fn_wav(batch):
    """ Collate function for AudioDataset
    """
    keys = []
    lengths = []
    feats_padded = []
    labels = []
    
    lengths = [ x[1].shape[0] for x in batch ]
    # max_len = max(max(lengths), 100)
    max_len = max(lengths)
    
    for i in range(len(batch)):
        keys.append(batch[i][0])
        act_len = lengths[i]
        pad_len = max_len - act_len
        feats_padded.append(np.pad(batch[i][1], ((0, pad_len), (0,0)), "constant"))

    if len(batch[0]) == 3:
        for i in range(len(batch)):
            labels.append(batch[i][2])
        return keys, torch.from_numpy(np.array(lengths)), torch.from_numpy(np.array(feats_padded)).float(), torch.from_numpy(np.array(labels))
    return keys, torch.from_numpy(np.array(lengths)), torch.from_numpy(np.array(feats_padded)).float()

if __name__=="__main__":
    #with open("conf/train_config_test.yml") as fid:
    with open("conf/train_config.yml") as fid:
        configs = yaml.load(fid, Loader=yaml.FullLoader)

            # Training data loader
    dataset_configs = configs['dataset']
    speed_perturb = dataset_configs['speed_perturb']
    feature_extraction_conf = dataset_configs['feature_extraction_conf']
    feature_dither = dataset_configs['feature_dither']
    spec_aug = dataset_configs['spec_aug']
    spec_aug_conf = dataset_configs['spec_aug_conf']       
    train_set = WavDataset("data/train/torch.scp",
                    with_label=True, 
                    shuffle=True,
                    feature_extraction_conf=feature_extraction_conf,
                    feature_dither=feature_dither,
                    speed_perturb=speed_perturb,
                    spec_aug=spec_aug,
                    spec_aug_conf=spec_aug_conf)
    train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True,
                    num_workers=6, collate_fn=collate_fn_wav, pin_memory=True)

    #print(len(FEATS_CACHE_DICT))
    #print("dataset 1 s_time:{}".format(datetime.datetime.now()))
    #for i in range(len(train_set)):
    #    train_set[i]
    #    pass
    #print("dataset 1 e_time:{}".format(datetime.datetime.now()))
    #print(len(FEATS_CACHE_DICT))
    print("dataloader 1 s_time:{}".format(datetime.datetime.now()))
    for batch_idx, (utt_ids, act_lens, data, target) in enumerate(train_loader):
        pass
    print("dataloader 1 e_time:{}".format(datetime.datetime.now()))
    print(len(FEATS_CACHE_DICT))
    #print("dataset 2 s_time:{}".format(datetime.datetime.now()))
    #for i in range(len(train_set)):
    #    train_set[i]
    #    pass
    #print("dataset 2 e_time:{}".format(datetime.datetime.now()))
    #print(len(FEATS_CACHE_DICT))


    print("dataloader 2 s_time:{}".format(datetime.datetime.now()))
    for batch_idx, (utt_ids, act_lens, data, target) in enumerate(train_loader):
        pass
    print("dataloader 2 e_time:{}".format(datetime.datetime.now()))
    print(len(FEATS_CACHE_DICT))

    #train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True,
    #                num_workers=6, collate_fn=collate_fn_wav)
    print("dataloader 3 s_time:{}".format(datetime.datetime.now()))
    for batch_idx, (utt_ids, act_lens, data, target) in enumerate(train_loader):
        pass
    print("dataloader 3 e_time:{}".format(datetime.datetime.now()))
    print(len(FEATS_CACHE_DICT))
