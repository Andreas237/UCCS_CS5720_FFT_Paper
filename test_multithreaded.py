import slovacek_fft as sf
from numpy import linspace
import time



n = 21
input = linspace(1,2**n,num=2**(n+1))





start_reg = time.time()
sf.sfft2(input)
end_reg = time.time()
regt = end_reg - start_reg


start_mt = time.time()
sf.sfft2mt(input)
end_mt = time.time()
mtt = end_mt - start_mt


print("Performance for %(samps)dsamples\nNon-threaded: %(nomt)f\nThreaded: %(mt)f" % {'samps':2**n, 'nomt':regt, 'mt':mtt})
