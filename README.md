# Data Science Project 4

## Anticipez les besoins en consommation électrique de bâtiments | Predict energy use and ghg emissions
### [Details](https://openclassrooms.com/fr/paths/164/projects/629/assignment)

Data from : [here](https://www.kaggle.com/datasets/city-of-seattle/sea-building-energy-benchmarking)

- Using Python 3.10.4 and virtual env
- librairies in requirements.txt (`pip install -r requirements.txt`)
- Mains notebooks are `nb_number_title`
- Notebooks not really relevant, not clean, used as guinea pigs are `nb_x+number_title`
- Tests are preceded or followed by `test` (duh.)
- Took the liberty of modifying shap to return a figure and actually work with non-default args. Doesnt mean I fixed more than I messed up. Executing the 
- code will need adjusted code on files mentionned below to perform as expected
- sources in sources.txt (duh)
- using python-dotenv to specify 
	- display PPI, default is forced to 100
	- Number of cpu cores (not used here, sklearn accepts parameters `n_jobs=-1` to use all cores)
- Unit testing is already implemented for a very limited amount of functions,  will try to expand the scope of the tests

<hr>

## Contents :

### 1 : cleaning = nb_00
### 2 : short analysis = nb_01
### 3 : feature engineering : nb_02 (though it's not actually doing anything atm)
### 4 : Modelisation of GHG : nb_03
### 5 : Modelisation of EUI : nb_04

<hr>

### Possible number 6 : Energy Star cert predict

<hr>
#### Side notes :

- Metadata read is informative but website (sources) is much more verbose.
- cleanme and lintme are basically junk files used to test code and check PEP8

<hr>


##### Using SHAP : Warning : Corrections on plots 

- changed pl by plt and used a plt.gcf() as a return value if show == False
- Concerns shap/plot/_waterfall.py, _violin.py and _beeswarm.py