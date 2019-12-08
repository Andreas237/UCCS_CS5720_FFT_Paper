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
from math import floor          # floor exponent in generateBasicSamples
import logging                  # stop using prints?
from datetime import date       # file naming




















def generateBasicSamples(power=10):

    """
    @brief Generate 2^power samples.
    @param power Number of samples to generate, default is 2^10=1024
    @return Samples generated for testing
    """
    return linspace(1,2**(floor(power/2)),num=2**power)



















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
    @param lin list of numic values
    @return The average of the list
    """
    return sum(lin) / len(lin)




























def verifyCorrectness(TESTS=10):
    """
    @brief Answering 'How did you validate the correctness of your
    implementation?'
    @method Generate TESTS sets of data.  Run my FFT versus SciPy FFT and compare
            the results.
    @param TESTS Number of data sets to compare FFTs against.  Default 10.
    """

    def compareArrays(array1,array2):
        """
        @brief Given two NumPy arrays, FFT results in this case, compare whether
               there is element-by-element equivalence.
        @param array1 numpy.array FFT results
        @param array2 numpy.array FFT results
        @return True/False
        """
        array1len = len(array1)
        array2len = len(array2)
        # Off the bat are they different size?
        if(array1len != array2len):
            logging.warn("""\tcompareArrays returned FALSE because the
                            inputs are of different length.""")
            # https://docs.python.org/3.8/howto/logging.html#logging-basic-tutorial
            return False

        else:
            # Compare i-th value of each array
            for i in range(array1len):
                if( round(array1[i],7) != round(array2[i],7)):
                    logging.warning("Index " + str(i) + "have values:\n" + str(array1[i]) + "\n" + str(array2[i]) )
                    return False
            return True
    # for TESTS
    #   1) Generate the data set
    #   2) Run my FFT versus the data set
    #   3) Run SciPy FFT versus the data set
    #   4) Compare results
    #       IF ERROR: count # of errors for this set versus size of set
    #       ELSE: Continue
    #   5) Repeat until TESTS is 0
    diff = True
    while(TESTS):
        samples = generateBasicSamples(TESTS) #1

        # 2,3,4
        if(compareArrays(sf.sfft2(samples),fft(samples)) != True):
            diff = False
            logging.warning("\t ARRAYS DIFFER FOR " + str(TESTS) + " samples.")
        else:
            logging.info("\t ARRAYS SAME  FOR " + str(TESTS) + " samples.")
        # 5
        TESTS -=1
    if(diff):
        print("""Successfully verified my FFT calcs the same answer as SciPy FFT to seven digits.""")
        logging.info("FFT validated against SciPy")
    else:
        logging.warning("FFT failed against SciPy")




























def verifyEfficiency(EXPONENT=1):
    """
    @brief Answering 'How did you test your implementation for
                      efficiency/effectiveness?'
    @method Run my FFT & SciPy FFT comparing runtimes until they differ by 20%.
            Log the size of the input data set for reference.
    @param EXPONENT Starting exponent for data set size.  2^EXPONENT is the size
           of the sample set.  Increment TESTS by 1 at each iteration.
    """


    def setupFile():
        """ Create a file for writing, set the header, and return its pointer.
            Name the file with todays date and the current time in milliseconds"""

        fout =  open('./Efficiency_Checks/check_at_' + str(date.today().strftime('%b-%d-%Y'))+ '_' + str(int(round(time.time() * 1000))) + '.csv','w+')
        fout.write("Samples,Samples_as_Exponents,My_Time,SciPy_Time,Ratio\n")
        return fout



    ratio = 1           # (my time)/(scipy time)
    beyond_ratio = 0    # After the ratio is exceeded check three more iterations
    fout = setupFile()  # Data file for logging results
    plot_data = ([],[],[])

    while(beyond_ratio <= 8):

        # 0 Dataset
        samples = generateBasicSamples(EXPONENT)

        # 1     Run my FFT and log runtime
        start_time = time.time()
        sf.sfft2(samples)
        end_time = time.time()
        my_time = end_time - start_time

        # 2     Run SciPy FFT and log runtime
        start_time = time.time()
        fft(samples)
        end_time = time.time()
        scipy_time = end_time - start_time

        # 3     Ratio =
        #               If mine is going 2x slower then cease after 5 iterations
        ratio = my_time / scipy_time
        if( ratio > 2 ):
            beyond_ratio += 1
        print("Samples: 2**{2}\t\t\tMy time: {0}\t\tSciPy time: {1}\t\tRatio: {3}\t\tBeyond Ratio: {4}".format(my_time,scipy_time,EXPONENT,ratio,beyond_ratio))

        # 4     Write the results to a file
        #       Format: Samples, Samples_as_Exponents, My_Time, SciPy_Time, Ratio
        fout.write("{0},2**{1},{2},{3},{4}\n".format(2**EXPONENT,EXPONENT,my_time,scipy_time,ratio))

        # 5 Increment Exponent
        EXPONENT += 1

        # 6 Save result for plotting
        plot_data[0].append(EXPONENT)
        plot_data[1].append(my_time)
        plot_data[2].append(scipy_time)

    # Close the file
    fout.close()

    return plot_data


































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
