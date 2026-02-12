from utilities import treatment, cot_estimator
import sys; sys.path.insert(0, "../../")
from dualbounds.generic import DualBounds

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from patsy import dmatrix

import pandas as pd

df= pd.read_csv('education.csv')

df_treated = df[df['treatment'] == True][['GPA', 'gpa_baseline']]
df_control = df[df['treatment'] == False][['GPA', 'gpa_baseline']]


def fit_Z(Z):
    # Fit KDE on original data
    kde = gaussian_kde(Z)
    
    simgap_Z = 0
    for _ in range(200):
        # Generate new Z samples with same size as original
        Z_sim = kde.resample(len(Z)).flatten()
        
        # Compute distributional differences
        simgap_Z += wasserstein_distance(Z, Z_sim)
           
    simgap_Z /= 200
    print(f'The wasserstein distance between simulated Z and original Z is: {simgap_Z:.3f}.')

    return kde


def spline_regression(Y, Z, Z_sampler, seed=123):
    np.random.seed(seed)
    
    z_min = Z.min()
    z_max = Z.max()
    Z_plot = np.linspace(z_min, z_max, 200).reshape(-1, 1)
    X_spline = dmatrix("bs(Z, df=6, degree=3, include_intercept=False)", {"Z": Z}, return_type='dataframe')
    X_plot_spline = dmatrix("bs(Z, df=6, degree=3, include_intercept=False)", {"Z": Z_plot.flatten()}, return_type='dataframe')
    
    # Cross-validation for Ridge regularization
    alphas = np.logspace(-4, 2, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_alpha = None
    best_score = np.inf
    
    for alpha in alphas:
        cv_scores = []
        for train_idx, val_idx in kf.split(X_spline):
            X_train, X_val = X_spline.iloc[train_idx], X_spline.iloc[val_idx]
            y_train, y_val = Y[train_idx], Y[val_idx]
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            cv_scores.append(mean_squared_error(y_val, preds))
        score = np.mean(cv_scores)
        if score < best_score:
            best_score = score
            best_alpha = alpha
    
    spline_model = Ridge(alpha=best_alpha)
    spline_model.fit(X_spline, Y)
    f_spline_est = spline_model.predict(X_plot_spline)
    
    # --- Plot Results ---
    plt.figure(figsize=(12, 5))
    plt.scatter(Z, Y, s=10, alpha=0.3, label='Observed Data')
    plt.plot(Z_plot, f_spline_est, color='red', label='Spline/GAM Estimate')
    plt.title('Spline/GAM Regression')
    plt.xlabel('Baseline GPA (Z)')
    plt.ylabel('Outcome GPA (Y)')
    plt.legend()
    plt.show()
    
    #compute the sigma of residual 
    y_pred = model.predict(X_spline)
    residuals = Y - y_pred
    
    # Estimate standard deviation (unbiased)
    n = X_spline.shape[0]
    p = X_spline.shape[1]  # number of spline basis functions (degrees of freedom)
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - p))
    

    simgap_Y = 0
    for _ in range(200):
        # Generate new Z samples with same size as original
        Z_sim = Z_sampler.resample(len(Z)).flatten()
        X_sim = dmatrix("bs(Z, df=6, degree=3, include_intercept=False)", {"Z": Z_sim}, return_type='dataframe')
        Y_sim = model.predict(X_sim)
        Y_sim += sigma_hat * np.random.randn(*Y_sim.shape)
        
        simgap_Y += wasserstein_distance(Y, Y_sim)
        
    
    simgap_Y /= 200
    print(f'The wasserstein distance between simulated Y and original Y is: {simgap_Y:.3f}.')
    
    
    return model, sigma_hat


def generate_data_model(
        N,
        Y_model0,
        Y_model1,
        std0,
        std1,
        Z_sampler
    ):
    
    # Generate N samples of Z 
    Z = Z_sampler.resample(N).reshape(-1,1)
    X = dmatrix("bs(Z, df=6, degree=3, include_intercept=False)", {"Z": Z}, return_type='dataframe')
    
    # Generate N samples of e ~ N(0, noise_std^2)
    Y0 = Y_model0.predict(X)
    e0 = std0 * np.random.randn(*Y0.shape) 
    Y0 += e0
    Y0 = Y0.reshape(-1,1)
    
    Y1 = Y_model1.predict(X)
    e1 = std1 * np.random.randn(*Y1.shape)  
    Y1 += e1
    Y1 = Y1.reshape(-1,1)

    return np.hstack((Y0, Y1, Z))


def trueval(
        f0, 
        f1,
        sigma0,
        sigma1,
        Z_sampler,
        SIM=500000,
        seed=123):
    np.random.seed(seed)
    Z_samples = Z_sampler(SIM)
    X_samples = dmatrix("bs(Z, df=6, degree=3, include_intercept=False)", {"Z": Z_samples}, return_type='dataframe')
    
    # Evaluate f0 and f1 on the samples
    diffs = f0.predict(X_samples) + f1.predict(X_samples)
    squared_norms = diffs**2
    expectation_term = np.mean(squared_norms)
    
    # Frobenius norm squared between covariance matrices
    frob_norm_sq = np.sum((sigma0 - sigma1)**2)
    
    return expectation_term + frob_norm_sq


#%%

def main_cp(N, tvc, f0, f1, std0, std1, Z_sampler, SIMSIZE=120):
    ridge_res = []
    knn_res = []
    cot_res = []
    
    np.random.seed(123)
    for k in range(SIMSIZE):
        # generate raw data 
        raw_data = generate_data_model(N,
                                       f0,
                                       f1,
                                       std0,
                                       std1,
                                       Z_sampler)
        
        # generate post-treatment data
        A0, A1, data = treatment(raw_data, 1, 1)
        
        A1[:, 0] = - A1[:, 0]
        # Direct Vc estimate
        cot = cot_estimator(A0, A1)
        cot_res.append(cot)
        
        # LL's estimate
        dbnd = DualBounds(
            f=lambda y0, y1, x: (y0 + y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='ridge',
        )
        
        results = dbnd.fit().results()
        LL_brd = results.at['Estimate', 'Lower']
        ridge_res.append(LL_brd)

        
        # LL's estimate3
        dbnd3 = DualBounds(
            f=lambda y0, y1, x: (y0 + y1) ** 2,
            covariates=data['X'],
            treatment=data['W'],
            outcome=data['y'],
            propensities=data['pis'],
            outcome_model='knn',
        )
        
        results3 = dbnd3.fit().results()
        LL_knn = results3.at['Estimate', 'Lower']
        knn_res.append(LL_knn)
        
        
        print(f'{k+1}/{SIMSIZE} complete.')
    
    
    return ridge_res, knn_res, cot_res

    
Z_sampler = fit_Z(np.array(df['gpa_baseline']))
Y_model1, std1 = spline_regression(np.array(df_treated['GPA']), np.array(df_treated['gpa_baseline']), Z_sampler)
Y_model0, std0 = spline_regression(np.array(df_control['GPA']), np.array(df_control['gpa_baseline']), Z_sampler)

tvc = trueval(Y_model0, Y_model1, std0, std1, Z_sampler)

# main run

sizelist = [400, 800, 1200, 1600]
ridge_reslist = {s: [] for s in sizelist}
knn_reslist = {s: [] for s in sizelist}
cot_reslist = {s: [] for s in sizelist}


for sz in sizelist:
    ridge_res, knn_res, cot_res = main_cp(sz, tvc, Y_model0, Y_model1, std0, std1, Z_sampler, SIMSIZE=500)
    ridge_reslist[sz].append(ridge_res)
    knn_reslist[sz].append(knn_res)
    cot_reslist[sz].append(cot_res)

import pickle

# Save to file
result_dicts = [ridge_reslist, knn_reslist, cot_reslist]

with open('result_dicts_plus.pkl', 'wb') as f:
    pickle.dump(result_dicts, f)


#%%Load the results
with open('result_dicts_plus.pkl', 'rb') as f:
    all_results = pickle.load(f)

#evaluate the relative error
tag = ['ridge', 'knn', 'adapt']
for i, dd in enumerate(all_results):
    for sz in dd:
        tmp = [v - tvc for v in dd[sz]]
        tmp = np.array(tmp)
        mm = np.mean(np.abs(tmp) / np.abs(tvc))
        vv = np.std(np.abs(tmp) / np.abs(tvc)) * (500 ** (-0.5))
        print(f'The mean relative error for {tag[i]} of size {sz} is: {mm:.4f}({vv:.4f}).')





