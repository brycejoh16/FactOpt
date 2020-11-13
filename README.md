# FactOpt
This is for a chen 5595 project. It is the optimization using machine learning of factory assembly line scheduling procedures. 


## Setup  
For this so far all you need is `tensorforce`,`numpy`, and `or_gym` 
so for now type in from terminal (mac) or anaconda command line: 

`conda create --name factopt  python=3.7.* numpy`

`conda activate factopt`

`pip install tensorforce or_gym`

Enviroment setup complete.
## knapsack

Current files: 
- `Knap_Sack_sorta_working` :knapsack v0 environment, utilizes GPU functionality for torch tensors. 
- `driver` :drive script to initilize stuff for a given run
- `environments`: where we can make various custom enviroments
- `reinforcement_learning`: top class which defines how reinforcement
is done for a specified agent and environment. 



