from cmath import exp, pi   # must use complex pi and exp
from numpy import array
import os, threading










################################################################################
def getAdds(x,w,ret,xLen):
    """
    Get the added terms
    """
    ret = [ x[i]+w[i] for i in range(xLen//2)]










################################################################################
def getSubs(x,w,ret,xLen):
    """
    Get the subtracted terms
    """
    ret = [ x[i]-w[i] for i in range(xLen//2)]





################################################################################
def sfft2(xIn):
    """
    Thank you wiki pseudocode https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
    and Carnegie Mellon for the concise walkthrough https://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/2001/pub/www/notes/fourier/fourier.pdf

    @param xIn Numpy Array
    @return r1+r1
    """


    # If only one element is given return it
    xLen = len(xIn)
    if xLen <= 1:   return xIn

    # Recursively split inputs for later butterflying
    Xeven = sfft2(xIn[0::2])   # even indexed elements
    Xodd = sfft2(xIn[1::2])   # odd indexed elements


    # Compute roots of unity/twiddle factors, O(N)
    w = [ Xodd[i] * exp(-2j*pi*i/xLen) for i in range(xLen//2)] # floor(xLen/2)
    # Re-combine the two halves, Twiddle factors computed once.
    r1 = [ Xeven[i]+w[i] for i in range(xLen//2)]   # O(N)
    r2 = [ Xeven[i]-w[i] for i in range(xLen//2)]   # O(N)
    return r1 + r2
# end def sfft2(xIn)
################################################################################







################################################################################
def sfft2mt(xIn):
    """
    Same as above, but multi-threaded
    https://www.geeksforgeeks.org/multithreading-in-python-set-2-synchronization/
    https://devarea.com/python-multitasking-multithreading-multiprocessing/
    Could try: http://ccom.uprrp.edu/~ikoutis/classes/algorithms_12/Lectures/MultithreadedAlgorithmsApril23-2012.pdf
    @param xIn Numpy Array
    @return r1+r1
    """


    # If only one element is given return it
    xLen = len(xIn)
    if xLen <= 1:   return xIn

    # Recursively split inputs for later butterflying
    Xeven = sfft2(xIn[0::2])   # even indexed elements
    Xodd = sfft2(xIn[1::2])   # odd indexed elements


    # Compute roots of unity/twiddle factors, O(N)
    w = [ Xodd[i] * exp(-2j*pi*i/xLen) for i in range(xLen//2)] # floor(xLen/2)
    # Re-combine the two halves
    #r1 = [ Xeven[i]+w[i] for i in range(xLen//2)]   # O(N)
    #r2 = [ Xeven[i]-w[i] for i in range(xLen//2)]   # O(N)
    r1 = r2 = []
    # create functions that handled adds and subtracts taking x,w,r1,r2
    t1 = threading.Thread(target=getAdds,args=(Xeven,w,r1,xLen))
    t2 = threading.Thread(target=getAdds,args=(Xeven,w,r2,xLen))
    # call threads on those functions
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    # test runtimes for each for 2**8

    return r1 + r2
