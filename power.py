"""Experiments on artificial data replicating part of the results in
Clemencon et al. (2009).
"""

import numpy as np
from sklearn.lda import LDA
from scipy.stats import ttest_ind # t-test for two independent samples
from sklearn.metrics import confusion_matrix
# from casella_moreno_2009 import log_B_10_fast
from sys import stdout


def truncated_multivariate_normal(mu, cov, size, a=-2.0, b=2.0):
    """Generate samples from N(mu,cov), then remove those outside
    [a,b] and then stack them till reaching size.
    """
    dataset = []
    tot = 0
    while tot < size :
        # print tot
        tmp = np.random.multivariate_normal(mu, cov=cov, size=size)
        # print tmp.shape
        tmp = tmp[np.logical_and(tmp>=a, tmp<=b).all(1), :]
        # print tmp.shape
        tot += tmp.shape[0]
        dataset.append(tmp)
        # print len(dataset)
    dataset = np.vstack(dataset)[:size,:]
    assert(dataset.shape[0]==size)
    return dataset

def two_sample(mu0, mu1, cov, m0, m1):
    """Generate two samples and return the stack of them.
    """
    # dataset_0 = np.random.multivariate_normal(mu0, cov=cov, size=m0)
    # dataset_1 = np.random.multivariate_normal(mu1, cov=cov, size=m1)
    dataset_0 = truncated_multivariate_normal(mu0, cov=cov, size=m0)
    dataset_1 = truncated_multivariate_normal(mu1, cov=cov, size=m1)
    X = np.vstack([dataset_0, dataset_1])
    return X

def build_Gamma(Gamma_diags):
    """Build Gamma (covariance) given the lists of diagonal values.
    """
    Gamma = np.zeros((4,4))
    for k, diag in enumerate(Gamma_diags):
        Gamma += np.diagflat(diag, k) # fill upper diagonals
        if k>0: Gamma += np.diagflat(diag, -k) # fill lower diagonals
    return Gamma


def compute_BF_10(cm, n_steps=10, M=10000):
    """Compute the Bayes factor given the model proposed by Casella
    and Moreno (2009) of the confusion matrix in order to test for the
    independence of predicted vs. true labels.
    """
    t_range = np.linspace(0, cm.sum(), n_steps).astype(np.int)
    BF_10 = np.zeros(len(t_range))
    for i, t in enumerate(t_range):
        log_B_10 = log_B_10_fast(cm, t=t, M=M)
        BF_10[i] = np.exp(log_B_10)
    return BF_10
    

if __name__ == '__main__':

    np.random.seed(0)

    experiment = 3 # The number of the experiment in Clemencon et al. (2009). See Section 5.
    p_threshold = 0.05 # p-value threshold. This is not written in the paper...
    BF_threshold = 2.5 # this should be the p_threshold equivalent. See Sellke and Berger...
    B = 150 # replications for computing power through simulation.

    # Create Gamma covariance matrix from diagonals taken from Clemencon et al. (2009).
    Gamma_diags_1 = [6.52, 3.84, 4.72, 3.1], [-1.89, 3.56, 1.52], [-3.2, 0.2], [-2.6]
    Gamma1 = build_Gamma(Gamma_diags_1)
    Gamma_diags = [1.83, 6.02, 0.69, 4.99], [-0.65, -0.31, 1.03], [-0.54, -0.03], [-1.24]
    Gamma = build_Gamma(Gamma_diags)
    
    # mean of the Gaussian in Clemencon et al. (2009).
    mu_1 = np.array([-0.96, -0.83,  0.29, -1.34])
    mu_2 = np.array([ 0.17, -0.24,  0.04, -1.02])
    mu_3 = np.array([ 1.19, -1.20, -0.02, -0.16])
    mu_4 = np.array([ 1.08, -1.18, -0.1,  -0.06])

    # Schema of the experiments in Clemencon et al. (2009):
    experiments_dict = {1: {'mu0': mu_1,
                            'mu1': mu_1,
                            'Gamma': Gamma1,
                            'm0': 500,
                            'm1': 500,
                            'n0': 500,
                            'n1': 500},
                        2: {'mu0': mu_1,
                            'mu1': mu_2,
                            'Gamma': Gamma1,
                            'm0': 500,
                            'm1': 500,
                            'n0': 500,
                            'n1': 500},
                        3: {'mu0': mu_3,
                            'mu1': mu_4,
                            'Gamma': Gamma,
                            'm0': 2000,
                            'm1': 1000,
                            'n0': 2000,
                            'n1': 1000},
                        4: {'mu0': mu_3,
                            'mu1': mu_4,
                            'Gamma': Gamma,
                            'm0': 3000,
                            'm1': 2000,
                            'n0': 3000,
                            'n1': 2000}}
    
    mu0 = experiments_dict[experiment]['mu0']
    mu1 = experiments_dict[experiment]['mu1']
    cov = experiments_dict[experiment]['Gamma']
    m0 = experiments_dict[experiment]['m0']
    m1 = experiments_dict[experiment]['m1']
    n0 = experiments_dict[experiment]['n0']
    n1 = experiments_dict[experiment]['n1']

    print("Experiment: %s" % experiment)
    print 'mu0:'  , mu0
    print 'mu1:'  , mu1
    print 'Gamma:', cov
    print 'm0:'   , m0 
    print 'm1:'   , m1 
    print 'n0:'   , n0 
    print 'n1:'   , n1 

    p_value = np.zeros(B)
    cm = []
    BF_10 = []
    for i in range(B):
        print i,
        stdout.flush()
        X_train = two_sample(mu0, mu1, cov, m0, m1)
        X_test = two_sample(mu0, mu1, cov, n0, n1)
        y_train = np.array([0]*m0 + [1]*m1)
        y_test = np.array([0]*n0 + [1]*n1)

        clf = LDA()
        clf.fit(X_train, y_train)
        delta_x = clf.transform(X_test) # distances from the classification surface.
        delta_x0 = delta_x[y_test==0]
        delta_x1 = delta_x[y_test==1]    
        t, p_value[i] = ttest_ind(delta_x0, delta_x1)
        # y_pred = clf.predict(X_test)
        # cm.append(confusion_matrix(y_test, y_pred))
        # BF_10.append(compute_BF_10(cm[-1]))
        
        # print p_value[i], p_value[1]<p_threshold

    print
    power = (p_value <= p_threshold).mean()
    print "LDA-Student Power =", power
    # BF_10 = np.vstack(BF_10)
    # power_BF = (BF_10.min(1) >= BF_threshold).mean()
    # print "BF Power =", power_BF
