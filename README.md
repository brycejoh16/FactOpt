# FactOpt
This is for a chen 5595 project. It is the optimization using machine learning of factory assembly line scheduling procedures. 


## Setup  

Type in from terminal (mac) or anaconda command line: 

`conda create --name <env>  python=3.7.* matplotlib numpy=1.18.5`

`conda activate <env>`

`pip install or_gym`

`conda install -c pytorch pytorch` 

Enviroment setup complete.
## knapsack

Current files: 
- `Knap_Sack_sorta_working` :knapsack v0 environment, utilizes GPU functionality for torch tensors. 
- `driver` :drive script to initilize stuff for a given run
- `environments`: where we can make various custom enviroments
- `reinforcement_learning`: top class which defines how reinforcement
is done for a specified agent and environment. 



