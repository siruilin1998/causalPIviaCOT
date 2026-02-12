import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

import numpy as np
from utilities import generate_data_prod, treatment, cot_estimator
from truevalue import true_vc_prod1dim
import matplotlib.pyplot as plt
seed = 123
np.random.seed(seed)
import statistics

b0 = np.array([[0.5]])
b1 = np.array([[1.1]])
k0 = np.array([[-0.35]])
k1 = np.array([[0.35]])
dy = 1
dz = 1 


def main_cp(N, tvc, SIMSIZE=120, gap=True):
    ridge_res = 0
    knn_res = 0
    cot_res = []

    for k in range(SIMSIZE):
        # generate raw data 
        raw_data = generate_data_prod(N, b0, b1, k0, k1)
        
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)

        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res.append(abs(cot - tvc))
        else:
            cot_res.append(cot)
            
        #LL's estimate
        dbnd = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='ridge',
        )
        
        results = dbnd.fit().results()
        LL_brd = results.at['Estimate', 'Lower']
        if gap:
            ridge_res += abs(LL_brd - tvc)
        else:
            ridge_res += LL_brd
        
        
        # LL's estimate3
        dbnd3 = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='knn',
        )
        
        results3 = dbnd3.fit().results()
        LL_knn = results3.at['Estimate', 'Lower']
        if gap:
            knn_res += abs(LL_knn - tvc)
        else:
            knn_res += LL_knn
        
        
        print(f'{k+1}/{SIMSIZE} complete.')
     
    ridge_res /= SIMSIZE
    knn_res /= SIMSIZE
    
    
    return ridge_res, knn_res, cot_res
    
    
# true Vc value 
tvc = true_vc_prod1dim(b0.item(), b1.item(), k0.item(), k1.item())

# main run
gap=True

sizelist = [200, 600, 1000, 1400, 1800]
ridge_reslist = []
knn_reslist = []
cot_reslist_mean = []
cot_reslist_std = []


for sz in sizelist:
    ridge_res, knn_res, cot_res = main_cp(sz, tvc, SIMSIZE=500, gap=gap)
    ridge_reslist.append(ridge_res)
    knn_reslist.append(knn_res)
    cot_reslist_mean.append(statistics.mean(cot_res))
    cot_reslist_std.append(statistics.stdev(cot_res))


plt.figure()
plt.plot(sizelist, np.array(ridge_reslist)/tvc, label='ridge', marker='o')
plt.plot(sizelist, np.array(knn_reslist)/tvc, label='knn', marker='o')
plt.plot(sizelist, np.array(cot_reslist_mean)/tvc, label='adapt', marker='o', color='purple')
# plt.errorbar(sizelist, np.array(cot_reslist_mean)/tvc, yerr=np.array(cot_reslist_mean)/tvc * (500)**(-0.5), fmt='-o', capsize=5, color='purple')

if not gap:
    plt.axhline(y=tvc, linestyle='--', label='true')

plt.xlabel('Sample size')
plt.ylabel('Average relative error')
plt.ylim(-0.01, 1.2)
plt.legend()
# plt.savefig("new500error_prod_error.pdf")
plt.show()

