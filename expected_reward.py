




import numpy as np

G=[]

'''
K : Number of items to sample
avg_cost: average cost of an item when sampled
max_weight: maximum weight allowed in box 
sampled_weight: highest weight allowed to sample 
'''
K=10
avg_cost=50
max_weight=200
sampled_weight=100
for _ in range(10000):
    G.append(np.min(np.random.random(K))*sampled_weight)


# note here ceiling is to account for
print('Expected average weight per box :%0.2f'%np.ceil(1/(K+1)*sampled_weight))
print('Average weight per box: %0.2f'%np.ceil(np.mean(G)))


# average cost we expect to see is 50
print('Highest possible mean reward: %0.2f'%(avg_cost * max_weight / np.ceil(np.mean(G))))



