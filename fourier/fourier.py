import numpy as np
import matplotlib.pyplot as plt
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)


# Define the square wave function
def square_wave(t):
    return  np.sign(np.sin(2.0 * np.pi * f0 * t))

def square_wave_torch(t):
    return torch.sign(torch.sin(2.0 * torch.pi * f0 * t))


# Fourier series approximation of the square wave
def square_wave_fourier(t, f0, N):
    result = np.zeros_like(t)
    for k in range(N):
        # The Fourier series of a square wave contains only odd harmonics.
        n = 2 * k + 1
        # Add harmonics to reconstruct the square wave.
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result

def square_wave_fourier_torch(t, f0, N):
    result = torch.zeros_like(t)
    for k in range(N):
        # The Fourier series of a square wave contains only odd harmonics.
        n = 2 * k + 1
        # Add harmonics to reconstruct the square wave.
        result += torch.sin(2 * torch.pi * n * f0 * t) / n
    return (4 / torch.pi) * result



def plot_square_wave(harmonics):
    plt.figure(figsize=(12, 8))
    # Plot the original square wave
    plt.subplot(2, 3, 1)
    plt.plot(t, square, 'k', label="Square wave")
    plt.title("Original Square Wave")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()
    # Plot Fourier reconstructions under different number of harmonics
    for i, Nh in enumerate(harmonics, start=2):
        plt.subplot(2, 3, i)
        y = square_wave_fourier(t, f0, Nh)
        plt.plot(t, y, label=f"N={Nh} harmonics")
        plt.plot(t, square, 'k--', alpha=0.5, label="Square wave")
        plt.title(f"Fourier Approximation with N={Nh}")
        plt.ylim(-1.5, 1.5)
        plt.grid(True)
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'fourier/square_wave_h={max(harmonics)}.png')



# 2. Apply the DFT and time the execution
def naive_dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.
    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).
    Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
    """
    N = len(x)
    # Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)
    # Iterate through each frequency bin (k)
    for k in range(N):
        # For each frequency bin, sum the contributions from all input samples (n)
        for n in range(N):
            # The core DFT formula: x[n] * e^(-2j * pi * k * n / N)
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X


def naive_dft_torch(x):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D signal that uses the GPU.
    Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
    """
    x = torch.tensor(x)
    # Ensure tensor is complex and on device
    x = x.to(device)
    if not torch.is_complex(x):
        x = x.to(torch.complex64)

    N = x.shape[0]

    # Indices
    n = torch.arange(N, device=device).reshape(1, -1)  # 1 * N
    k = torch.arange(N, device=device).reshape(-1, 1)  # N * 1

    # k*n gives a matrix of the indices of the omegas in F
    # DFT formula
    F = torch.exp(-2j * torch.pi * k * n / N)

    # Use matrix multiplication instead of loop to get GPU benefitss
    X = F @ x

    return X


def high_harmonics():
    # List of harmonic numbers used to construct the square wave
    h1 = [1, 3, 5]
    h2 = [1, 5, 9, 13, 17]
    h3 = [1, 13, 25, 47, 49]
    plot_square_wave(h1)
    plot_square_wave(h2)
    plot_square_wave(h3)


# Set parameters for the signal
# N = 2 ** 11 # Number of sample points
# N = 2 ** 12
N = 2 ** 13

print(f'Number of sample points: {N}')

T = 1.0 # Duration of the signal in seconds
f0 = 1 # Fundamental frequency of the square wave in Hz

# Create the time vector
# np.linspace generates evenly spaced numbers over a specified interval.
# We use endpoint=False because the interval is periodic.
t = np.linspace(0, T, N)
# Generate the original square wave
square = square_wave(t)

# Construct a square wave using 50 harmonics
signal = square_wave_fourier(t, f0, 50)

# 2. Apply the DFT and Time the Execution
# Time the naïve DFT implementation
start_time_naive = time.time()
dft_result = naive_dft(signal)
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive

#  Time NumPy's FFT implementation
start_time_fft = time.time()
fft_result = np.fft.fft(signal)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft

# time pytorch naive dft
start_time_tdft = time.time()
tdft_result = naive_dft_torch(signal)
tdft_result = tdft_result.cpu().numpy()
end_time_tdft = time.time()
tdft_duration = end_time_tdft - start_time_tdft

# 3. Print Timings and Verification
print("--- DFT/FFT Performance Comparison ---")
print(f"Naive DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
print(f"Pytorch DFT Execution Time: {tdft_duration:.6f} seconds")

# It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")
    
# Check if our implementation is close to NumPy's result
# np.allclose is used for comparing floating-point arrays.
print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")

# 4. Prepare for Plotting
# Generate the frequency axis for the plot.
# np.fft.fftfreq returns the DFT sample frequencies.
# We only need the first half of the frequencies (the positive ones) due to symmetry.
xf = np.fft.fftfreq(N, d=T/N)[:N//2]
# We normalize the magnitude by N and multiply by 2 to get the correct amplitude.
magnitude = 2.0/N * np.abs(dft_result[0:N//2])

def plot_results():
    print('plotting results...')
    # 5. Visualize the Results
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the original time-domain signal
    ax1.plot(t, signal, color='c')
    ax1.set_title('Input Sine Wave Signal', fontsize=16)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_xlim(0, 1.0) # Show a few cycles of the sine wave
    ax1.grid(True)

    # Plot the frequency-domain signal (magnitude of the DFT)
    ax2.stem(xf, magnitude, basefmt=" ")
    ax2.set_title(
    f'Discrete Fourier Transform (Magnitude Spectrum). N={N}',
    fontsize=16
    )
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_xlim(0, 50) # Focus on lower frequencies
    ax2.grid(True)

    # Add vertical lines for the first ten frequencies
    for i in range(20):
        if i < len(xf) and i % 2 == 1: # Only plot odd harmonics
            ax2.axvline(
            xf[i], color='r', linestyle='--', alpha=0.7,
            label=f'f{i}: {i}* f0 = {xf[i]:.1f} Hz'
        )
            
    # Only show labels for first 3 frequencies to avoid cluttering
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f'fourier/fourier_results_N={N}.png')

plot_results()
