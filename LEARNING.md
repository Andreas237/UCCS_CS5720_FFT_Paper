# Learning #
Here are some tidbits I learned while working on this project.

 * `matplotlib.plot.sublot`: When you build a `subplot(X,Y,Z)` you have *X* x *Y*
    places to fill in your subplot frame.  Top-to-bottom and left-to-right they
    are indexed from *1* to *X* x *Y*.  *Z* indicates which spot your placing
    a plot in.
 * Multiprocessing/Multithreading: Algorithms need to be designed for mp/mt to
   be good candidates.  I have yet to take a class discussing either of these
   topics, but some of the resources I came across made the point.  My mt test
   did not yield any useful speedup.  Maybe it is my implementation or ineptitude.
   More studying will tell.
 * Welch's method takes sample data and applies windows to discretize the data,
   then returns the frequencies, times and spectrogram of the data.
 * Spectrogram with PSD: windowing the data is important.  Welch's method would
   is designed for Periodogram however works here to that that a smaller set
   of points is analyzed (windows).



### Repository Pages ###
 1. [Some of the notes on what I learned working on this project](https://github.com/Andreas237/UCCS_CS5720_FFT_Paper/blob/master/LEARNING.md)
 2. [Resources referenced while working on this project](https://github.com/Andreas237/UCCS_CS5720_FFT_Paper/blob/master/RESOURCES.md)
 3. [Readme](https://github.com/Andreas237/UCCS_CS5720_FFT_Paper/blob/master/README.md)
