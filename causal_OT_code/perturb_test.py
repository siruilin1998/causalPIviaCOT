import sys; sys.path.insert(0, "../../")

import numpy as np
import matplotlib.pyplot as plt
from utilities import generate_data_perturb, generate_data_square_perturb, generate_data_prod_perturb, treatment, cot_estimator
from truevalue import true_vip1dim, true_vc1dim, true_vu1dim, true_vc_square, true_vc_prod1dim
import statistics

seed = 123
np.random.seed(seed)

b0 = np.array([[-0.6]])
b1 = np.array([[1.6]])
dy = 1
dz = 1

def main_pt(N, eps, SIMSIZE=120, gap=True):
    cot_res1 = []
    cot_res2 = []
    cot_res3 = []

    b0 = np.array([[-0.6]])
    b1 = np.array([[1.6]])
    tvc1 = true_vc1dim(-b0.item(), b1.item())
    
    
    b0 = np.array([[-0.2]])
    b1 = np.array([[0.6]])
    tvc2 = true_vc_square(-b0, b1)
    
    b0 = np.array([[0.5]])
    b1 = np.array([[1.1]])
    k0 = np.array([[-0.35]])
    k1 = np.array([[0.35]])
    tvc3 = true_vc_prod1dim(b0.item(), b1.item(), k0.item(), k1.item())

    for k in range(SIMSIZE):
        # generate raw data 
        b0 = np.array([[-0.6]])
        b1 = np.array([[1.6]])
        raw_data = generate_data_perturb(N, eps, b0, b1)
        
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)
        
        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res1.append(abs(cot - tvc1))
        else:
            cot_res1.append(cot)
        
        
        b0 = np.array([[-0.2]])
        b1 = np.array([[0.6]])
        raw_data = generate_data_square_perturb(N, eps, b0, b1)
    
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)
        
        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res2.append(abs(cot - tvc2))
        else:
            cot_res2.append(cot)
            
        b0 = np.array([[0.5]])
        b1 = np.array([[1.1]])
        k0 = np.array([[-0.35]])
        k1 = np.array([[0.35]])
        
        raw_data = generate_data_prod_perturb(N, eps, b0, b1, k0, k1)
        
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, dy, dz)

        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        if gap:
            cot_res3.append(abs(cot - tvc3))
        else:
            cot_res3.append(cot)
        
        
        print(f'{k+1}/{SIMSIZE} complete.')
    
    
    return cot_res1, tvc1, cot_res2, tvc2, cot_res3, tvc3


# main run: 
gap=True #record estimation error (True) or estimation value (False)
N = 1500
epslist = np.linspace(0, 0.1, 5)
reslist1 = []
reslist2 = []
reslist3 = []

SIM = 500

for eps in epslist:
    cot_res1, tvc1, cot_res2, tvc2, cot_res3, tvc3 = main_pt(N, eps, SIMSIZE=SIM, gap=gap)
    reslist1.append((statistics.mean(cot_res1), statistics.stdev(cot_res1) * SIM**(-0.5)))
    reslist2.append((statistics.mean(cot_res2), statistics.stdev(cot_res2) * SIM**(-0.5)))
    reslist3.append((statistics.mean(cot_res3), statistics.stdev(cot_res3) * SIM**(-0.5)))

reslist1 = np.array(reslist1) / tvc1

reslist2 = np.array(reslist2) / tvc2

reslist3 = np.array(reslist3) / tvc3


plt.figure()
plt.plot(epslist, [a[0] for a in reslist1], label='adapt (linear)', marker='^')
plt.plot(epslist, [a[0] for a in reslist2], label='adapt (quadratic)', marker='^')
plt.plot(epslist, [a[0] for a in reslist3], label='adapt (scale)', marker='^', color = 'purple')


plt.xlabel('Perturbation scale')
plt.ylabel('Average relative error')
plt.ylim(-0.01, 0.2)
plt.legend()
# plt.savefig("new500error_perturb_error.pdf")
plt.show()


