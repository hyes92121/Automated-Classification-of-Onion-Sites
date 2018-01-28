# Project Overview
A project to classify Onion Sites.

# Steps:
1. run getBOW.py with the first command line argument as a file containing all urls of onion sites and the second arg as the output directory name.
ex. 
	python3 getBOW.py url.csv wrdgroups
2. run preprocess.py with the first command line argument as the same directory name as specified above.
ex. 
	python3 preprocess.py wrdgroups
3. run predict.sh 
ex.
	sh predict.sh 

##  How to run .ipynb (IPython notebook) files
`pip3 install --upgrade pip`

`pip3 install jupyter`

`jupyter notebook`

Press `Shift+Enter` to execute a cell block






