# %% z square 1-dim
from truevalue import true_vc_square
from utilities import generate_data_square, treatment, cot_estimator
import numpy as np
from dualbounds.generic import DualBounds
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../../")
import statistics


seed = 123
np.random.seed(seed)

b0 = np.array([[-0.2]])
b1 = np.array([[0.6]])
dy = 1
dz = 1


def main_cp(N, tvc, SIMSIZE=120, gap=True):
    ridge_res = 0
    knn_res = 0
    cot_res = []
    for k in range(SIMSIZE):
        # generate raw data
        raw_data = generate_data_square(N, b0, b1)
    
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)
        
        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res.append(abs(cot - tvc))
        else:
            cot_res.append(cot)
    
        # LL-ridge estimate
        dbnd_brd = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='ridge',
        )
    
        result_brd = dbnd_brd.fit().results()
        LL_brd = result_brd.at['Estimate', 'Lower']
        if gap:
            ridge_res += abs(LL_brd - tvc)
        else:
            ridge_res += LL_brd
        
    
        # LL-knn estimate
        dbnd_knn = DualBounds(
            f=lambda y0, y1, x: (y0 - y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='knn',
        )
    
        result_knn = dbnd_knn.fit().results()
        LL_knn = result_knn.at['Estimate', 'Lower']
        if gap:
            knn_res += abs(LL_knn - tvc)
        else:
            knn_res += LL_knn
    
        print(f'{k+1}/{SIMSIZE} complete.')
    
    ridge_res /= SIMSIZE
    knn_res /= SIMSIZE
    # cot_res /= SIMSIZE
    
    return ridge_res, knn_res, cot_res


# true Vc value
tvc = true_vc_square(-b0, b1)

# main run
gap=True  #record estimation error (True) or estimation value (False)

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
plt.ylim(-0.01, 0.8)
plt.legend()
# plt.savefig("new500error_quad_error.pdf")
plt.show()

