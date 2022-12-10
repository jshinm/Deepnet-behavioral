# Deepnet-behavioral
Online behavioral experiment for deepnet paper powered by Flask

<strike>https://ib-behavior.herokuapp.com/</strike>

The above website no longer available as heroku stopped free services as of Nov 28, 2022

<!-- TODO: -->

## Installation
```
pip install -r requirements.txt
```

## Running Local Server
```
python main.py
```

## Logs

02/04/21 <br>
- [x] add codes for analytic pipeline

01/29/21 <br>
- [x] debug: sprial contracting issue
- [x] sample size reduced to 100
- [x] increase range from (-2,2) to (-3,+3)
- [x] number of trial increased from 50 to 100
- [x] add functionality to download sql db
- [x] minor adjustments to figure styling

01/08/21 <br>
- [x] debug: check for memory leakage (switched to "agg" backend for matplotlib)
- [x] connect to heroku postgres for database management (sqlite for local, postgres for production)

12/23/20 <br>
- [x] add "catch" trials (only sampled within the unit circle; interspersed throughout the test)
- [x] add practice trials at the end of the tutorials (currently 5 trials)
- [x] include consent language (placed in the very first page)
- [x] implement spiral posterior
- [x] spiral parameters modified (reduced noise 2.5 -> 1.0)
- [x] add feedback panel at the end of the experiment
- [ ] <strike>populate exactly the equal parts of each pattern</strike>

12/17/20 <br>
- [x] modify tutorial to emphasize likelihood judgement
- [x] further simplify tutorials and visually more attractive
- [x] remove slider bubble and change tick labels

12/09/20 <br>
- [x] Hard set the experiment trials to 50
- [x] Simplify tutorials
- [x] Create manual admin mode to activate/deactivate admin panels
- [x] Change slider style
- [x] Change overall interface styling
- [x] Update the manual
- [x] Have someone to run through the experiment
- [x] Admin mode deactivated for deployment