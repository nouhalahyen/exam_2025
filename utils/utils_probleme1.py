from os.path import join, isdir, isfile
from os import listdir as ls
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter1d

KERNEL = torch.tensor(np.concatenate([np.arange(0.025,0.5,0.025), np.array([1]), np.zeros(19)]), dtype=torch.float32)

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
    # Simulate parent events
    expected_parents = h * duration
    parent_events = np.random.exponential(1/h, int(np.ceil(expected_parents)))
    parent_times = np.cumsum(parent_events)
    parent_times = parent_times[parent_times < duration]

    offspring_times = []

    for parent_time in parent_times:
        # Number of offspring for each parent
        num_offspring = np.random.poisson(theta)
        # Offspring times relative to parent
        offspring_delays = np.random.randn(num_offspring) * p
        offspring_event_times = parent_time + offspring_delays
        # Filter offspring times to keep only those within the duration
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
    # Simulate parent events
    expected_parents = h * duration
    parent_events = np.random.exponential(1 / h, int(np.ceil(expected_parents)))
    parent_times = np.cumsum(parent_events)
    parent_times = parent_times[parent_times < duration]

    offspring_times = []

    for parent_time in parent_times:
        # Number of offspring for each parent
        num_offspring = np.random.poisson(theta)
        # Offspring times relative to parent
        offspring_delays = np.random.randn(num_offspring) * p
        offspring_event_times = parent_time + offspring_delays
        # Filter offspring times to keep only those within the duration
        offspring_times.extend(offspring_event_times[(offspring_event_times >= 0) & (offspring_event_times <= duration)])

    return np.sort(parent_times), np.sort(np.array(offspring_times))


from scipy.ndimage import gaussian_filter1d
def smooth_events_with_gaussian_window(event_times, duration, sigma, resolution=1):
    """
    Smooth event occurrences over time using a Gaussian window.

    Parameters:
    - event_times: A numpy array of event times.
    - duration: The total duration of the simulation, in seconds.
    - sigma: The standard deviation for the Gaussian kernel, controls the smoothing.
    - resolution: The time resolution of the simulation (default is 1 second).

    Returns:
    - A tuple containing the time series and the smoothed event density.
    """
    # Create a time series of event occurrences
    time_series_length = int(duration / resolution)
    event_series = np.zeros(time_series_length)

    # Mark the occurrences of events in the time series
    for time in event_times:
        if time < duration:
            index = int(time / resolution)
            event_series[index] += 1

    # Apply Gaussian window smoothing
    smoothed_series = gaussian_filter1d(event_series, sigma=sigma/resolution, mode='constant')

    # Generate time points for the x-axis
    times = np.arange(0, duration, resolution)

    return times, smoothed_series



def convert(x):
  return 0.1*x**0.7

# @torch.jit.script
def apply_conv1d_with_custom_kernel(signal, kernel):
    """
    Apply a 1D convolution to a signal with a custom kernel using PyTorch.

    Parameters:
    - signal: A 1D PyTorch tensor representing the input signal.
    - kernel: A 1D PyTorch tensor representing the custom convolutional kernel.
    - stride: The stride of the convolution. Defaults to 1.

    Returns:
    - The convolved signal as a 1D PyTorch tensor.
    """
    # Ensure the signal and kernel are properly formatted for conv1d
    signal = signal.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and out_channels dimensions

    # Calculate padding to maintain output size
    padding = (kernel.size(-1) - 1) // 2

    # Apply the 1D convolution
    convolved_signal = F.conv1d(signal, kernel, padding=padding)

    # Remove added dimensions to return to original signal shape
    convolved_signal = convolved_signal.squeeze(0).squeeze(0)

    return convolved_signal

# @torch.jit.script
def generate_random_sinusoidal_process(length, mean_poisson=10., phase_range=(0, 2 * np.pi)):
    """
    Generate a signal composed of a polynomial of sinusoids with random phases, coefficients, and periods.

    Parameters:
    - length: The number of points in the generated signal.
    - mean_poisson: The mean of the Poisson distribution for determining the number of components.
    - phase_range: The range (min, max) of possible phases for the sinusoids.
    - time_scale: Scale for the time axis (default 1.0).

    Returns:
    - A PyTorch tensor representing the generated signal.
    """
    # Determine the number of sinusoidal components
    num_components = 1 + int(torch.poisson(torch.tensor(mean_poisson)).item())

    # Sample coefficients from a normal distribution
    coefficients = torch.randn(num_components)

    # Sample periods from a uniform distribution between 120 and 24*60
    periods = torch.randint(low=240, high=24*60, size=[num_components]).float()

    # Time axis
    t = torch.linspace(0, length, steps=length)

    # Initialize the signal
    signal = torch.zeros(length)

    # Generate each sinusoidal component
    for i in range(num_components):
        # Random phase for each component
        phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

        # Sinusoidal component
        component = coefficients[i] * torch.sin(2 * np.pi * t / periods[i] + phase)

        # Add the component to the signal
        signal += component
    signal /= num_components
    signal /= 2
    return signal


def get_gaussian_noise(signal, noise_scale_func):
    """
    Apply Gaussian noise to a signal where the noise standard deviation varies non-linearly with the signal intensity.

    Parameters:
    - signal: A PyTorch tensor representing the input signal.
    - noise_scale_func: A function defining how the noise standard deviation scales with the signal intensity.

    Returns:
    - The signal with applied Gaussian noise.
    """
    # Compute the standard deviation of the noise for each point in the signal
    noise_std = noise_scale_func(signal)

    # Generate the Gaussian noise
    noise = torch.randn_like(signal) * noise_std

    return noise

# Define a non-linear noise scale function
def noise_scale_func(signal_intensity):
    # Example: Standard deviation of the noise scales with the square of the signal intensity
    return 0.1*torch.abs(1 + signal_intensity) ** 0.75



def generate_pair(duration, distance, dtheta, dp):
    # ppp_intensity is inversely proportional to the distance

    ppp_intensity = 0.05 * distance
    theta = 10. + dtheta
    p = 3. + dp
    _, event_times = simulate_neyman_scott_process_1d(h=ppp_intensity, theta=10., p=3., duration=duration)

    times, smoothed_series = smooth_events_with_gaussian_window(event_times, duration=duration, sigma=2)
    ground_truth_specific = torch.tensor(smoothed_series, dtype=torch.float32)
    ground_truth = ground_truth_specific * 1/distance

    converted_smoothed_series = convert(smoothed_series)
    converted_smoothed_series = torch.tensor(converted_smoothed_series, dtype=torch.float32)
    convolved_smoothed_series = apply_conv1d_with_custom_kernel(converted_smoothed_series, KERNEL)

    lf_noise = generate_random_sinusoidal_process(duration)
    hf_noise = get_gaussian_noise(convolved_smoothed_series, noise_scale_func)

    noisy1_series = convolved_smoothed_series + lf_noise
    noisy_series = noisy1_series + hf_noise

    return ground_truth, noisy_series


#################################################################################################################
#################################################################################################################
#############################################      DATASET       ################################################
#################################################################################################################
#################################################################################################################




# Step 1: Define the Dataset Class
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
        dtheta = idx/1000.
        dp = 1 - idx/1000.
        ground_truth, noisy_series = generate_pair(self.duration, dist, dtheta, dp)

        # normalisation:
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

        # Normalize and apply margins
        noisy_series = 0.3 * noisy_series[self.margin:-self.margin]
        ground_truth = ground_truth[self.margin:-self.margin]

        return idx, dist, ground_truth, noisy_series


# Step 2: Create the DataLoader
def create_dataloader(duration, idx2distance, batch_size, shuffle=True):
    dataset = TensorPairDataset(duration, idx2distance)
    # num_workers = 2 for colab
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader

# Step 2: Create the DataLoader
def create_dataloader_with_margin(duration, idx2distance, batch_size, shuffle=True, margin=120):
    dataset = TensorPairDatasetWithMargin(duration, idx2distance, margin=margin)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader
