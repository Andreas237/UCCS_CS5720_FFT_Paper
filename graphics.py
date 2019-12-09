import numpy as np
import matplotlib.pyplot as plt         # For plotting
from experiments import *               # Experiments generate plot data
import slovacek_fft as sf               # My FFT
from scipy.fftpack import fft           # optimized FFT for comparison
from scipy import signal                # Spectrogram and Periodogram
from scipy.io import wavfile            # Read a WAV file
import random                           # random number generator for large input comparison

























################################################################################
def plotter():
    """
    Example plot from https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    """
    t = np.linspace(0, 0.5, 500)
    s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    plt.plot(t, s)
    plt.show()

# End def plotter()
################################################################################



























################################################################################
def fftPlotter():
    """
    Example plot from https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
    """
    t = np.linspace(0, 0.5, num=128)
    s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)

    fft = sf.sfft2(s)

    T = t[1] - t[0]  # sampling interval
    N = s.size

    # 1/T = frequency
    f = np.linspace(0, 1 / T, N)

    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    # Real signal symmetry only requires half the samples
    plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
    plt.show()

# End def fftPlotter()
################################################################################



























################################################################################
def plotAddedSineWaves():
    """
    Add three sign waves 20,90,110 Hz together and show that waveform
    """
    # time = np.linspace(0,1,2**9)
    time = np.linspace(0,.5,num=512)
    waves =   np.sin(50 * 2 * np.pi * time) \
            + np.sin(110 * 2 * np.pi * time) \
            + np.sin(175 * 2 * np.pi * time)

    # Plot 1: What multiple waves together look like
    plt.subplot(1,2,1)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.plot(time, waves)

    # Extracting composite frequencies
    fft = sf.sfft2(waves)
    Interval = time[1] - time[0] # interval in which to get samples
    Samples = waves.size
    frequency = np.linspace(0,1/Interval,Samples) # f = 1/T
    plt.subplot(1,2,2)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(frequency[:Samples//2], np.abs(fft)[:Samples//2], width=1.5)
    #plt.plot(frequency[:Samples//2], np.abs(fft)[:Samples//2] * 1 / Samples)

    plt.show()

# End def plotAddedSineWaves()
################################################################################



























################################################################################
def plotBasicExperiment():
    """
    Plot the results of my basic experiment, defined in experiments.py
    """
    mine, optimized = basicExperiment(4,10)


    plt.plot(*zip(*mine.items()),label='My FFT',color='r',marker='8',linestyle=':')
    plt.plot(*zip(*optimized.items()),label='Optimized FFT',color='b',marker='d',linestyle='-.')
    plt.xlabel('N where Samples to Process is 2^N')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime of Fast Fourier Transform')
    plt.grid()      # grid for approximate k-value
    plt.legend()    # turns on labels in legend
    plt.show()

# End def plotBasicExperiment()
################################################################################



























################################################################################
def plotCompareDensities():
    """
    Compare a spectrogram of dense points and sparse points with both FFTs
    """

    # 1 Generate "dense" data
    dense_samples = 2**12
    dense_time = np.linspace(0,2,num=dense_samples)
    dense_interval = dense_time[1] - dense_time[0]          # interval in which to get samples
    dense_waves =   np.sin(400 * 2 * np.pi * dense_time) \
                    + np.sin(433 * 2 * np.pi * dense_time)

    # 2 Generate "sparse" data
    sparse_samples = 2**8
    sparse_time = np.linspace(0,2,num=sparse_samples)
    sparse_interval = sparse_time[1] - sparse_time[0]          # interval in which to get samples
    sparse_waves =   np.sin(400 * 2 * np.pi * sparse_time) \
                    + np.sin(433 * 2 * np.pi * sparse_time)


    # 3 Calculate plot inputs
    dense_scp = fft(dense_waves)
    dense_me = sf.sfft2(dense_waves)
    dense_frequency = np.linspace(0,1/dense_interval,dense_samples) # f = 1/T

    sparse_scp = fft(sparse_waves)
    sparse_me = sf.sfft2(sparse_waves)
    sparse_frequency = np.linspace(0,1/sparse_interval,sparse_samples) # f = 1/T



    # Dense verus sparse data requires 3 rows and 2 columns
    # plt.subplot(3,2,1)

    """Plot the added frequency waves"""
    plt.subplot(3,2,1)
    plt.title("Dense Data Spectrogram")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.plot(dense_time, dense_waves)
    plt.grid(True)


    plt.subplot(3,2,2)
    plt.title("Sparse Data Spectrogram")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.plot(sparse_time, sparse_waves)
    plt.grid(True)






    """
    Plot the SciPy FFTs
    """
    plt.subplot(3,2,3)
    plt.title("SciPy FFT Applied to Dense Data")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(dense_frequency[:dense_samples//2], np.abs(dense_scp)[:dense_samples//2], width=1.5,color='tab:orange')
    plt.grid(True)


    plt.subplot(3,2,4)
    plt.title("SciPy FFT Applied to Sparse Data")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(sparse_frequency[:sparse_samples//2], np.abs(sparse_scp)[:sparse_samples//2], width=1.5,color='tab:orange')
    plt.grid(True)






    """
    Plot the My FFTs
    """
    plt.subplot(3,2,5)
    plt.title("My FFT Applied to Dense Data")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(dense_frequency[:dense_samples//2], np.abs(dense_me)[:dense_samples//2], width=1.5,color='tab:blue')
    plt.grid(True)


    plt.subplot(3,2,6)
    plt.title("My FFT Applied to Sparse Data")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(sparse_frequency[:sparse_samples//2], np.abs(sparse_me)[:sparse_samples//2], width=1.5,color='tab:blue')
    plt.grid(True)






    plt.tight_layout()
    plt.show()

# end plotCompareDensities
################################################################################



























################################################################################
def plotCompareSpectrumResults():
    """
    Check if my FFT gets the same result as SciPy when applied to actual inputs.
    Add three sign waves 20,90,110 Hz together and show the waveforms
    """

    time = np.linspace(0,.5,num=512)
    waves =   np.sin(50 * 2 * np.pi * time) \
            + np.sin(110 * 2 * np.pi * time) \
            + np.sin(175 * 2 * np.pi * time)

    plt.subplot(3,1,1)

    # Plot the sine wave before analysis
    plt.ylabel("Amplitude")
    plt.xlabel("Time (seconds)")
    plt.title("Before analysis")
    plt.subplot(3,1,1)
    plt.plot(time, waves)
    plt.grid(True)



    # Plot SciPy versus the input
    scpfft = fft(waves)
    Interval = time[1] - time[0] # interval in which to get samples
    Samples = waves.size
    frequency = np.linspace(0,1/Interval,Samples) # f = 1/T
    plt.subplot(3,1,2)
    plt.title("SciPy FFT")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(frequency[:Samples//2], np.abs(scpfft)[:Samples//2], width=1.5,color='tab:orange')
    plt.grid(True)



    # Plot Mine versus the input
    myfft = sf.sfft2(waves)
    Interval = time[1] - time[0] # interval in which to get samples
    Samples = waves.size
    frequency = np.linspace(0,1/Interval,Samples) # f = 1/T
    plt.subplot(3,1,3)
    plt.title("My FFT")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency (hertz)")
    # Real signal symmetry only requires half the samples
    plt.bar(frequency[:Samples//2], np.abs(myfft)[:Samples//2], width=1.5,color='tab:blue')
    plt.grid(True)


    plt.tight_layout()
    plt.show()

# End def plotCompareSpectrumResults()
################################################################################






























################################################################################
def plotVerifyEfficiency():
    """
    Plot the results of the speed test in experiments.py/verifyEfficiency().
    """


    # plot_data[0] - exponent for generating samples, i.e. 2**EXP
    # plot_data[1] - My FFT runtime for that sample set
    # plot_data[2] - SciPy FFT runtime for that sample set
    plot_data = verifyEfficiency()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(plot_data[0], plot_data[1], color='tab:blue',label='My FFT')   # My times
    ax.plot(plot_data[0], plot_data[2], color='tab:orange',label='SciPy FFT') # SciPy times
    ax.set_ylabel("Runtime in Seconds")
    ax.set_xlabel("Number of Samples, 2^X")

    fig.suptitle("FFT Runtimes versus growing sample sets")
    ax.legend()
    plt.show()

# End def plotVerifyEfficiency()
################################################################################






























################################################################################
def plotWavAnalysis():
    """
    Read .WAV files, then plot the spectrogram with power-spectral density.
    Refer to: https://stackoverflow.com/questions/44787437/how-to-convert-a-wav-file-to-a-spectrogram-in-python3

    Method:
        Spectrogram and Periodogram want to generate their own frequencies.
        Attempt 1 I'll substitute mine with theirs.

        Failed to use mine since it isn't windowed.
    """
    # Cut to the last ~30 seconds of the song since it had the most interesting
    # spectrogram.
    rate, data = wavfile.read('./Give_Love_A_Try.wav')
    f, t, Sxx = signal.spectrogram(data,rate)

    #my_f = sf.sfft2(data)
    #plt.pcolormesh(t, nf, np.log(Sxx))


    plt.pcolormesh(t, f, np.log(Sxx))
    plt.title("Cut of 'Give Love A Try,' by Unknown - Spectrogram with PSD")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# End def plotWavAnalysis()
################################################################################






























################################################################################
def plotWavAnalysisDeep():
    """
    Read .WAV files, then plot a full analysis of the signal as in:
    https://matplotlib.org/gallery/lines_bars_and_markers/spectrum_demo.html#sphx-glr-gallery-lines-bars-and-markers-spectrum-demo-py

    Plot 1:
    """
    # Cut to the last ~30 seconds of the song since it had the most interesting
    # spectrogram.
    rate, data = wavfile.read('./Give_Love_A_Try.wav')
    f, t, Sxx = signal.spectrogram(data,rate)

    myfft = sf.sfft2(f)
    Interval = t[t.size-1] - t[0]
    Fs = len(myfft)                     # no. of samples
    s = np.abs(myfft)[:len(myfft)//2]   # Real signal symmetry only requires half the samples


    # plot time signal:
    plt.subplot(3,2,1)
    plt.title("Signal")
    plt.plot(s , color='C0')
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # plot different spectrum types:
    plt.subplot(3,2,3)
    plt.title("Magnitude Spectrum")
    plt.magnitude_spectrum(s, Fs=Fs, color='C1')


    plt.subplot(3,2,4)
    plt.title("Logarithmic Magnitude Spectrum")
    plt.magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')


    plt.subplot(3,2,5)
    plt.title("Phase Spectrum")
    plt.phase_spectrum(s, Fs=Fs, color='C2')


    plt.subplot(3,2,6)
    plt.title("Anlge Spectrum")
    plt.angle_spectrum(s, Fs=Fs, color='C2')
    plt.tight_layout()
    plt.show()

# End def plotWavAnalysisDeep()
################################################################################
