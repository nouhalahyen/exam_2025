from os.path import join, isdir, isfile
from os import listdir as ls
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter1d

KERNEL = torch.tensor(np.concatenate([np.arange(0.025, 0.5, 0.025), np.array([1]), np.zeros(19)]), dtype=torch.float32)

def simulate_neyman_scott_process_1d(h, theta, p, duration):
    """
    Simulate a 1D Neyman-Scott process as a function of time.

    Parameters:
    - h: Intensity of the Poisson process for parent events.
    - theta: Mean of the Poisson distribution for the number of offspring per parent event.
    - p: Standard deviation of the Gaussian distribution for the timing of offspring events relative to their parent.
    - duration: Duration of the process.

    Returns:
    - parent_times: A list of event times for parent events.
    - offspring_times: A list of event times for all offspring events.
    """
    expected_parents = h * duration
    parent_events = np.random.exponential(1 / h, int(np.ceil(expected_parents)))
    parent_times = np.cumsum(parent_events)
    parent_times = parent_times[parent_times < duration]

    offspring_times = []

    for parent_time in parent_times:
        num_offspring = np.random.poisson(theta)
        offspring_delays = np.random.randn(num_offspring) * p
        offspring_event_times = parent_time + offspring_delays
        offspring_times.extend(offspring_event_times[(offspring_event_times >= 0) & (offspring_event_times <= duration)])

    return np.sort(parent_times), np.sort(offspring_times)


@jit(nopython=True)
def simulate_neyman_scott_process_1d_jit(h, theta, p, duration):
    """
    Simulate a 1D Neyman-Scott process as a function of time.

    Parameters:
    - h: Intensity of the Poisson process for parent events.
    - theta: Mean of the Poisson distribution for the number of offspring per parent event.
    - p: Standard deviation of the Gaussian distribution for the timing of offspring events relative to their parent.
    - duration: Duration of the process.

    Returns:
    - parent_times: A sorted array of event times for parent events.
    - offspring_times: A sorted array of event times for all offspring events.
    """
    expected_parents = h * duration
    parent_events = np.random.exponential(1 / h, int(np.ceil(expected_parents)))
    parent_times = np.cumsum(parent_events)
    parent_times = parent_times[parent_times < duration]

    offspring_times = []

    for parent_time in parent_times:
        num_offspring = np.random.poisson(theta)
        offspring_delays = np.random.randn(num_offspring) * p
        offspring_event_times = parent_time + offspring_delays
        offspring_times.extend(offspring_event_times[(offspring_event_times >= 0) & (offspring_event_times <= duration)])

    return np.sort(parent_times), np.sort(np.array(offspring_times))


def smooth_events_with_gaussian_window(event_times, duration, sigma, resolution=1):
    """
    Smooth event occurrences over time using a Gaussian window.
    """
    time_series_length = int(duration / resolution)
    event_series = np.zeros(time_series_length)

    for time in event_times:
        if time < duration:
            index = int(time / resolution)
            event_series[index] += 1

    smoothed_series = gaussian_filter1d(event_series, sigma=sigma / resolution, mode='constant')
    times = np.arange(0, duration, resolution)

    return times, smoothed_series


def convert(x):
    return 0.1 * x ** 0.7


def apply_conv1d_with_custom_kernel(signal, kernel):
    """
    Apply a 1D convolution to a signal with a custom kernel using PyTorch.
    """
    signal = signal.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    padding = (kernel.size(-1) - 1) // 2
    convolved_signal = F.conv1d(signal, kernel, padding=padding)
    return convolved_signal.squeeze(0).squeeze(0)


def generate_random_sinusoidal_process(length, mean_poisson=10., phase_range=(0, 2 * np.pi)):
    """
    Generate a signal composed of a polynomial of sinusoids with random phases, coefficients, and periods.
    """
    num_components = 1 + int(torch.poisson(torch.tensor(mean_poisson)).item())
    coefficients = torch.randn(num_components)
    periods = torch.randint(low=240, high=24 * 60, size=[num_components]).float()
    t = torch.linspace(0, length, steps=length)
    signal = torch.zeros(length)

    for i in range(num_components):
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]
        signal += coefficients[i] * torch.sin(2 * np.pi * t / periods[i] + phase)

    signal /= num_components
    signal /= 2
    return signal


def get_gaussian_noise(signal, noise_scale_func):
    """
    Apply Gaussian noise to a signal where the noise standard deviation varies non-linearly with the signal intensity.
    """
    noise_std = noise_scale_func(signal)
    noise = torch.randn_like(signal) * noise_std
    return noise


def noise_scale_func(signal_intensity):
    return 0.1 * torch.abs(1 + signal_intensity) ** 0.75


def generate_pair(duration, distance, dtheta, dp):
    """
    Generate a pair of ground truth and noisy data based on the Neyman-Scott process and noise additions.
    """
    ppp_intensity = 0.05 * distance
    theta = 10. + dtheta
    p = 3. + dp
    _, event_times = simulate_neyman_scott_process_1d(h=ppp_intensity, theta=10., p=3., duration=duration)
    times, smoothed_series = smooth_events_with_gaussian_window(event_times, duration=duration, sigma=2)
    ground_truth_specific = torch.tensor(smoothed_series, dtype=torch.float32)
    ground_truth = ground_truth_specific * 1 / distance
    converted_smoothed_series = convert(smoothed_series)
    converted_smoothed_series = torch.tensor(converted_smoothed_series, dtype=torch.float32)
    convolved_smoothed_series = apply_conv1d_with_custom_kernel(converted_smoothed_series, KERNEL)
    lf_noise = generate_random_sinusoidal_process(duration)
    hf_noise = get_gaussian_noise(convolved_smoothed_series, noise_scale_func)
    noisy1_series = convolved_smoothed_series + lf_noise
    noisy_series = noisy1_series + hf_noise
    return ground_truth, noisy_series


class TensorPairDataset(Dataset):
    def __init__(self, duration, idx2distance):
        self.duration = duration
        self.idx2distance = idx2distance
        self.num_cmls = len(idx2distance)
        self.indices = sorted(idx2distance.keys())

    def __len__(self):
        return self.num_cmls

    def __getitem__(self, i):
        idx = self.indices[i]
        dist = self.idx2distance[idx]
        dtheta = idx / 1000.
        dp = 1 - idx / 1000.
        ground_truth, noisy_series = generate_pair(self.duration, dist, dtheta, dp)
        noisy_series = 0.3 * noisy_series
        return idx, dist, ground_truth, noisy_series


class TensorPairDatasetMargin(Dataset):
    def __init__(self, duration, idx2distance, margin=120):
        self.duration = duration
        self.idx2distance = idx2distance
        self.num_cmls = len(idx2distance)
        self.indices = sorted(idx2distance.keys())
        self.margin = margin

    def __len__(self):
        return self.num_cmls

    def __getitem__(self, i):
        idx = self.indices[i]
        dist = self.idx2distance[idx]
        dtheta = idx / 1000.
        dp = 1 - idx / 1000.
        ground_truth, noisy_series = generate_pair(self.duration, dist, dtheta, dp)
        noisy_series = 0.3 * noisy_series[self.margin:-self.margin]
        ground_truth = ground_truth[self.margin:-self.margin]
        return idx, dist, ground_truth, noisy_series


def create_dataloader(duration, idx2distance, batch_size, shuffle=True):
    dataset = TensorPairDataset(duration, idx2distance)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader


def create_dataloader_with_margin(duration, idx2distance, batch_size, shuffle=True, margin=120):
    dataset = TensorPairDatasetMargin(duration, idx2distance, margin=margin)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader
