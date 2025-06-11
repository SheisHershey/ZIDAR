# ZIDAR

This is the source code for ZIDAR.

## Requirements
python                    3.7

torch                     >=1.8.0

gluonts                   >=0.9.0

numpy                     1.16

pandas                    1.1

## Tips

The function *distributions* in *gluonts* needs to be replaced by the identically named .zip file which we offer in main branch.

You can reproduce our results by directly run the python file main.py.

We offer a zip format of processed M5 data in file data, you need to unzip it before running our example.

## Concrete Implementing Ways
### step 1
Download all the files to your local directory.
### step 2
Unzip the *salestv_data.zip* in the folder *data*.
### step 3
Replace the command line 184 in main.py by your local directory. Run it.
