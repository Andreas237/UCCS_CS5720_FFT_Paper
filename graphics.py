import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from experiments import *
import slovacek_fft as sf


























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
    plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
    plt.show()




























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
    plt.bar(frequency[:Samples//2], np.abs(fft)[:Samples//2], width=1.5)
    #plt.plot(frequency[:Samples//2], np.abs(fft)[:Samples//2] * 1 / Samples)

    plt.show()




























def plotVerifyEfficiency():
    """Plot the results of the speed test in experiments.py/verifyEfficiency()."""
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
