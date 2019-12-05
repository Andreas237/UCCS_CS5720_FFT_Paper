# This file contains experiments and their helper functions.
#
#
#
# generateSamples()     builds lists of inputs for
#
#
#
# Basic Evauluation:
#   Hypothesis: my FFT will not work as well as an optimized FFT when inputs are
#               large.
#   Method:     Run my FFT versus SciPy's FFT for increasingly large inputs.
#               1) Generate several inputs: 2^[10:50]
#
#
#

from numpy import linspace      # generate inputs
import slovacek_fft as sf       # my FFT
from scipy.fftpack import fft   # optimized FFT for comparison
import time                     # measure times




















def generateSamples(start=10,end=60):

    """
    @brief Generate a dictionary of lists.
                    Key:  exponent to which 2 is raise, 2^key is the number of
                          elements in the list
                    Val:  the list of elements
    Use NumPy linspace to generate a list of samples, and associate that with
    the exponent used to generate the samples.
    @param start Number of samples to generate, default is 2^10=1024
    @param end End point for number of samples to generate, default is 2^60
    @return samples Generated data for testing
    """

    # Verify that the bounds are valid
    if(start > end): raise Exception("start must be less than end")


    # Use linspace to generate samples from 1 to 2^n, with a spacing of 2^(n+1)
    # linspace(1,2**n,1/(2**(1.5*n)))
    # add this to the dictionary
    samples = {}
    samples = { n: linspace(1,2**n,num=2**(n+1)) for n in range(start,end) }
    # return the dictionary
    return samples




















def averageTime(lin):
    """
    @brief Given a list of numbers calculate the average
    @ param lin list of numic values
    @ return The average of the list
    """
    return sum(lin) / len(lin)




























def basicExperiment(start=10,end=11):
    """
    Run Slovacek FFT 10 times for every sample.
    """

    # Generate sample inputs
    samples = generateSamples(start,end)

    # How many measures to take versus each sample
    measure_count = 10


    # Initialize dictionaries to capture key=n (2^n samples) versus average time
    # for FFT computation
    slovacek_data = {}
    scipy_data = {}
    # Measure Slovacek FFT vs. SciPy FFT
    for k,v in samples.items():
        # Lists to capture runtime for each FFT
        slovacek_times = []
        optimized_times = []

        print("Running experiments for key ",k," of ",end," with ",2**k," samples.")
        # Run each FFT multiple times
        for i in range(measure_count):

            # Time Slovacek FFT
            start_time = time.time()
            sf.sfft2(v)
            end_time = time.time()
            slovacek_times.append(end_time-start_time)

            # Times for SciPy FFT
            start_time = time.time()
            fft(v)
            end_time = time.time()
            optimized_times.append(end_time-start_time)

        # Add the data points to the dictionary
        slovacek_data[k] = averageTime(slovacek_times)
        scipy_data[k] = averageTime(optimized_times)

    return slovacek_data, scipy_data
