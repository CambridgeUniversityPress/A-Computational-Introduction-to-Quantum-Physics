This repository contains complete solutions to all exercises involving numerical implementations in the book "A Computational Introduction to Quantum Physics".

All source code is self-contained; every task is solved within a specific file. As a consequence, there is a lot of redundancy. Many files have several identical lines of code. 
We have tried to write transparent and well-documented code.

Two exercises involves reading files with input data. These data files are also included in the relevant directory.

Source code is provided both for Python and for MATLAB/Octave - in their respectice main directories. Beyond these directories, all code is separated in directories 
corresponding to chapters in the book. 

Several implementations involves simulating quantum systems on the fly, in which case the evolution is shown directly - rather than generating movies to be studied afterwards. 
This should work fine for the MATLAB implementations, but when it comes to Python, it may become a bit machine- and platform dependent to what extent this works. 
The scripts have primarily been developed within the Spyder environment on Linux, where it seems to work fine - provieded that the Graphics backend is set to "Authomatic" rather 
than "Inline". It should also work to run it directly from the terminal.

In case you find errors or things which are unreasonable cumbersome, the author would be most appreciative to your input.
