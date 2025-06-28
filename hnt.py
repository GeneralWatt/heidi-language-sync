# Import everything
import matplotlib.pyplot as plt
import numpy as np

import ffmpeg_wrap
import signal_tools as st
import fft

import os
import shutil
import sys
import ffmpeg
from PIL import Image
import imagehash
import cv2
import torch
from torchvision import models, transforms
from scipy.signal import correlate
from scipy.spatial.distance import *
from fastdtw import fastdtw
from dtaidistance import dtw
from scipy.signal import savgol_filter
from scipy import signal

from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def hamming_2d(shape):
    #This function make a 2d hamming window (fading circle)
    #Print Image of blob to see how it looks
    i,j = shape
    a = np.r_[[np.hamming(i)]*j].T
    b = np.c_[[np.hamming(j)]*i]
    return a*b

def blob_generator(shape):
    #This function generates a mask over the video.
    #The mask is a grid of four fading circles.
    #Print Image of blob to see how it looks
    i,j = shape
    ii = int(i//2); jj = int(j//2)
    ham2d = hamming_2d([ii,jj])
    arr = np.zeros([i,j])
    arr[:ii,:jj] = ham2d
    arr[:ii,-jj:] = -ham2d
    arr[-ii:,:jj] = -ham2d
    arr[-ii:,-jj:] = ham2d
    return arr

def roundint(input):
    return np.round(input).astype('int')

def normalize_signal(v):
    """
    Normalize a signal by removing the mean and dividing by the 90th percentile.

    Parameters:
        v (array-like): Input signal (list or NumPy array)

    Returns:
        np.ndarray: Normalized signal
    """
    v = np.array(v, dtype=np.float64)
    v -= np.mean(v)
    v /= np.percentile(v, 90)
    return v

def import_2D_video_signal(filename,shape):
    #Read video and output into 2D frames using ffmepg frame reader.
    frames = []
    with ffmpeg_wrap.FfmpegFrameReader(filename, shape) as r:
        f = r.get_next_frame()
        while f is not None:
            frames.append(f)
            f = r.get_next_frame()

    return frames

def convert_2_1D_signal_blob(frames,blob_mask):
    '''
    This function converts the 2D frames into a 1D video signal by applying the blob mask to the video
    by taking the areas of the blob pattern into consideration
    '''

    v = []
    for f in frames:
        v.append(np.sum(blob_mask.T*f[:,:,0]))

    return normalize_signal(v)

def convert_2_1D_signal_phash(frames):
    '''
    This function converts the 2D (image) frames into a 1D video signal by converting
    each frame into a 64-bit hash and then convert that into a int
    '''
    frame_signals= []
    for i in frames:
        img = Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        h = imagehash.phash(img)  # returns 64-bit hash
        frame_signals.append(int(str(h), 16))  # convert to integer
    return normalize_signal(np.array(frame_signals))

def convert_2_1D_signal_mpi(frames):
    '''
    This function converts the 2D (image) frames into a 1D video signal by converting
    each frame into a 64-bit hash and then convert that into a int
    '''
    frame_signals= []
    for i in frames:
        gray = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))
        value = gray.mean()
        frame_signals.append(value)
    return normalize_signal(np.array(frame_signals))

def convert_2_1D_signal_cnn(frames):
    '''
    This function uses AI
    '''
    # 1. Load pretrained ResNet-18
    model = models.resnet18(pretrained=True)
    model.eval()  # Disable training mode

    # 2. Define transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    scalars = []
    for i in frames:
        color_img = cv2.cvtColor(i, cv2.COLOR_YUV2RGB)
        img_pil = Image.fromarray(color_img)

        input_tensor = preprocess(img_pil).unsqueeze(0)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            features = model(input_tensor)  # Shape: [1, 1000] (logits)

        # 5. Reduce to scalar (option A: vector norm)
        scalar_value = features.norm(p=2).item()
        scalars.append(scalar_value)

    return np.array(scalars)

def simple_slice_maker(xlen, wsize, overlap_frac=0.1):
    """
    Simplified version of slice_maker.

    Parameters
    ----------
    xlen : int
        Length of signal to slice.
    wsize : int
        Window size for each slice.
    overlap_frac : float
        Fraction of window to overlap (e.g. 0.1 = 10% overlap).

    Yields
    ------
    (i, j): tuple
        Start and end indices of window.
    dampwin: np.ndarray
        Cosine window taper of length (j - i).
    """
    overlap = int(wsize * overlap_frac)
    step = wsize - overlap
    lovlap = overlap // 2
    rovlap = overlap - lovlap

    # Cosine taper
    curve = np.cos(np.linspace(np.pi, 0, overlap)) * 0.5 + 0.5

    for start in range(0, xlen, step):
        i = max(0, start - lovlap)
        j = min(xlen, start + wsize + rovlap)
        winlen = j - i

        dampwin = np.ones(winlen)
        if i > 0:
            dampwin[:overlap] *= curve
        if j < xlen:
            dampwin[-overlap:] *= curve[::-1]

        yield i, j, dampwin

def fft_phase_offset(sig1, sig2, upsample=1):
    """
    Estimate the time delay (offset) between two signals using FFT phase correlation.

    Parameters
    ----------
    sig1 : np.ndarray
        Reference signal (1D).
    sig2 : np.ndarray
        Signal to align (1D), same length as sig1.
    upsample : int, optional
        If >1, upsamples the cross-correlation for sub-sample accuracy.

    Returns
    -------
    offset : float
        Estimated delay: sig2 should be shifted by this amount to align with sig1.
    correlation : float
        Maximum correlation value (normalized).
    """
    N = len(sig1)
    if len(sig2) != N:
        raise ValueError("Signals must be the same length")

    # Remove mean
    sig1 = sig1 - np.mean(sig1)
    sig2 = sig2 - np.mean(sig2)

    # FFTs
    f1 = np.fft.fft(sig1)
    f2 = np.fft.fft(sig2)

    # Cross-power spectrum
    R = f1 * np.conj(f2)
    R /= np.abs(R) + 1e-8  # Normalize to avoid magnitude bias

    # Inverse FFT to get cross-correlation
    cc = np.fft.ifft(R).real
    cc = np.fft.fftshift(cc)

    # Upsample (optional, for sub-sample accuracy)
    if upsample > 1:
        cc = np.interp(
            np.linspace(0, N-1, N * upsample),
            np.arange(N),
            np.fft.fftshift(cc)
        )
        max_idx = np.argmax(cc)
        offset = max_idx / upsample - N // 2
    else:
        max_idx = np.argmax(cc)
        offset = max_idx - N // 2

    # Normalize correlation (optional)
    correlation = cc[max_idx] / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-8)

    return offset, correlation

def normal_correlation(slice1, slice2):
    # Compute cross-correlation
    corr = correlate(slice1, slice2, mode='full', )
    lags = np.arange(-len(slice1) + 1, len(slice2))

    best_shift = lags[np.argmax(corr)]

    return -best_shift

def bilateral_1d(y, sigma_spatial=5, sigma_intensity=0.1):
    N = len(y)
    smoothed = np.zeros_like(y)
    for i in range(N):
        weights = np.exp(-0.5 * ((np.arange(N)-i)**2 / sigma_spatial**2
                                + (y - y[i])**2 / sigma_intensity**2))
        weights /= weights.sum()
        smoothed[i] = np.sum(weights * y)
    return smoothed

    smoothed = bilateral_1d(signal)