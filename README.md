This repository contains complete solutions to all exercises involving numerical implementations in the book "A Computational Introduction to Quantum Physics".

All source code is self-contained; every task is solved within a specific file. As a consequence, there is a lot of redundancy. Many files have several identical lines of code. 
We have tried to write transparent and well-documented code.

Two exercises involve reading files with input data. These data files are also included in the relevant directory.

Source code is provided both for Python and for MATLAB/Octave - in their respective main directories. Beyond these directories, all code is separated in directories 
corresponding to chapters in the book. 

Several implementations involve simulating quantum systems on the fly, in which case the evolution is shown directly - rather than generating movies to be studied afterwards. 
This should work fine for the MATLAB implementations, but when it comes to Python, it may become a bit machine- and platform dependent to what extent this works. 
The scripts have primarily been developed within the Spyder environment on Linux, where it seems to work fine - provided that the Graphics backend is set to "Automatic" rather 
than "Inline". It should also work to run it directly from the terminal.

In case you find errors or things which are unreasonably cumbersome, the author would be most appreciative to your input.

When solving the exercises, do note that you need to be aware and cautions about the input parameters, which - alsmost exclusively - have been hard coded. 
Some of them relate to the physical system at hand, others relate to the numerics. In case of the latter, do note that these source files come with no guarantee that the numerical precision is adequate. The user must check this - by running and re-running the code with improved numerics until the results are virtually independent of the numerical parameters.
