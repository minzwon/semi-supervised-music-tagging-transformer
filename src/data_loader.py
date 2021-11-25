# coding: utf-8
import os
import random
import pickle
import torch
import numpy as np
import soundfile as sf
from torch.utils import data
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)


class MSDDataset(data.Dataset):
    def __init__(self, data_path, data_split, num_samples, num_chunks, is_augmentation):
        assert data_split in ['TRAIN', 'VALID', 'TEST', 'STUDENT', 'NONE']
        self.data_path = data_path
        self.data_split = data_split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self._load_data()
        if is_augmentation or (data_split=='STUDENT'):
            self._get_augmentations()

    def __getitem__(self, index):
        wav = self._read_audio(index)
        if self.data_split == 'TRAIN':
            wav = self._process_train(wav)
            binary = self.binaries[index].astype('float32')
            return wav, binary
        elif self.data_split in ['VALID', 'TEST']:
            wav = self._process_valid(wav)
            binary = self.binaries[index].astype('float32')
            return wav, binary
        elif self.data_split == 'STUDENT':
            original_wav, noisy_wav = self._process_student(wav)
            return original_wav, noisy_wav

    def _load_data(self):
        split_fn = os.path.join(self.data_path, 'splits', '%s_ids.npy' % self.data_split.lower())
        self.track_ids = np.load(split_fn)
        if self.data_split in ['TRAIN', 'VALID', 'TEST']:
            binary_fn = os.path.join(self.data_path, 'splits', '%s_binaries.npy' % self.data_split.lower()) 
            self.binaries = np.load(binary_fn)

    def _get_augmentations(self):
        # Stochastic data augmentation
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8),
            RandomApply([Delay(sample_rate=22050)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4),
            RandomApply([Reverb(sample_rate=22050)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _read_audio(self, index):
        # get audio path
        track_id = self.track_ids[index]
        filename = '{}/{}/{}/{}.wav'.format(track_id[2], track_id[3], track_id[4], track_id)
        audio_path = os.path.join(self.data_path, 'audio', filename)

        # read audio
        wav, _ = sf.read(audio_path)

        # downmix to mono
        if len(wav.shape) == 2:
            wav = np.mean(wav, axis=1)
        return wav

    def _process_train(self, wav):
        # Randomly crop a short chunk.
        random_index = random.randint(0, len(wav) - self.num_samples - 1)
        wav = wav[random_index : random_index + self.num_samples].astype('float32')
        if self.is_augmentation:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
        return wav

    def _process_valid(self, wav):
        # Take multiple chunks and stack them. They will be averaged later after prediction to make a song-level prediction.
        hop = (len(wav) - self.num_samples) // self.num_chunks
        wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)]).astype('float32')
        return wav

    def _process_student(self, wav):
        # One original audio and one noised audio.
        random_index = random.randint(0, len(wav) - self.num_samples - 1)
        original_wav = wav[random_index : random_index + self.num_samples].astype('float32')
        noisy_wav = self.augmentation(torch.from_numpy(original_wav).unsqueeze(0)).squeeze(0).numpy()
        return original_wav, noisy_wav

    def __len__(self):
        return len(self.track_ids)


def msd_dataloader(data_path, data_split, batch_size, num_samples, num_workers, num_chunks, is_augmentation=False):
    data_loader = data.DataLoader(dataset=MSDDataset(data_path, data_split, num_samples, num_chunks, is_augmentation),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader

