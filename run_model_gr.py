import numpy as np
from astropy.io import ascii
from functions import *



print("loading data")
cols = ['g_r',
        'flag_vsquared',
        'flag_voidfinder']
data = ascii.read('data_flags_updated.dat', include_names=cols)
g_r = data['g_r']

print("loading classification")
#classification V^2
wall_v2 = np.argwhere(data['flag_vsquared'] == 0).flatten()
void_v2 = np.argwhere(data['flag_vsquared'] == 1).flatten()
edge_v2 = np.argwhere(data['flag_vsquared'] == 2).flatten()
out_v2 = np.argwhere(data['flag_vsquared'] == 9).flatten()

#classification voidfinder
wall_vf = np.argwhere(data['flag_voidfinder'] == 0).flatten()
void_vf = np.argwhere(data['flag_voidfinder'] == 1).flatten()
edge_vf = np.argwhere(data['flag_voidfinder'] == 2).flatten()
out_vf = np.argwhere(data['flag_voidfinder'] == 9).flatten()

## property variables
print("loading property model variables")

# data and range
data1_wall_vf = remove_nan(g_r[wall_vf])
data2_void_vf = remove_nan(g_r[void_vf])
data1_wall_v2 = remove_nan(g_r[wall_v2])
data2_void_v2 = remove_nan(g_r[void_v2])
bins_ = np.linspace(-0.5,1.5, 250)
label_vf = 'g-r_VF'
label_v2 = 'g-r_V2'

# Model 1

#minimizer bounds
bounds1 = [[0.1, 10.],     # s ........ scale factor (peak 1 to 2) #2.5

           [50., 5000.],  # a ........ skew normal 1 amplitude # 2000 and 100
           [0.0, 0.75],     # mu_a ..... skew normal 1 location  # 0.5
           [0.1, 1.5],     # sigma_a .. skew normal 1 scale #0.2
           [-10., 10.],    # a_skew ... skew normal 1 skew

           [50., 7000.],  # b ........ skew normal 2 amplitude #6100 and 150
           [0.75, 1.5],     # mu_b ..... skew normal 2 location #0.9  # should not overlap with mu_a
           [0.05, 1.5],     # sigma_b .. skew normal 2 scale #0.1
           [-10., 10.]]    # b_skew ... skew normal 2 skew


def prior_xform1(u):
    """Priors for the 9 parameters of model M1. Required by the dynesty sampler.

    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.

    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    s       = uniform(0.1, 10., u[0])
    a       = jeffreys(1., 1e4, u[1])
    mu_a    = uniform(0., 0.75, u[2])
    sigma_a = jeffreys(0.1, 2., u[3])
    xi_a    = uniform(-10., 10., u[4])
    b       = jeffreys(1., 1e4, u[5])
    mu_b    = uniform(0.75, 2., u[6])
    sigma_b = jeffreys(0.05, 2., u[7])
    xi_b    = uniform(-10., 10., u[8])

    return s, a, mu_a, sigma_a, xi_a, b, mu_b, sigma_b, xi_b


#Model 2

# minimizer bounds
bounds1_ = [[100., 5000.],  # a ........ skew normal 1 amplitude #2000
        [0.0, 0.7],     # mu_a ..... skew normal 1 location #0.5
        [0.1, 2.0],     # sigma_a .. skew normal 1 scale #0.7
        [-10., 10.],    # a_skew ... skew normal 1 skew
        [100., 5000.],  # b ........ skew normal 2 amplitude #6100
        [0.7, 1.5],     # mu_b ..... skew normal 2 location #0.9
        [0.1, 2.0],     # sigma_b .. skew normal 2 scale #0.4
        [-10., 10.]]    # a_skew ... skew normal 2 skew

bounds2_ = [[100., 1000.],   # g ........ skew normal 3 amplitude #200
            [0.0, 0.7],     # mu_g ..... skew normal 3 location #0.5
            [0.1, 2.0],     # sigma_g .. skew normal 3 scale #0.6
            [-10., 10.],    # g_skew ... skew normal 3 skew
            [100., 1500.],   # d ........ skew normal 4 amplitude #800
            [0.7, 1.5],     # mu_d ..... skew normal 4 location #0.8
            [0.05, 2.0],     # sigma_d .. skew normal 4 scale #0.3
            [-10., 10.]]    # d_skew ... skew normal 4 skew
bounds2 = bounds1_ + bounds2_

def prior_xform2(u):
    """Priors for the 9 parameters of model M1. Required by the dynesty sampler.

    Parameters
    ----------
    u : ndarray
        Array of uniform random numbers between 0 and 1.

    Returns
    -------
    priors : ndarray
        Transformed random numbers giving prior ranges on model parameters.
    """
    a       = jeffreys(1., 1e4, u[0])
    mu_a    = uniform(0., 0.7, u[1])
    sigma_a = jeffreys(0.1, 2., u[2])
    xi_a    = uniform(-10., 10., u[3])
    b       = jeffreys(1., 1e4, u[4])
    mu_b    = uniform(0.7, 1.5, u[5])
    sigma_b = jeffreys(0.05, 2., u[6])
    xi_b    = uniform(-10., 10., u[7])

    g       = jeffreys(1., 1e4, u[8])
    mu_g    = uniform(0., 0.7, u[9])
    sigma_g = jeffreys(0.1, 2., u[10])
    xi_g    = uniform(-10., 10., u[11])
    d       = jeffreys(1., 1e4, u[12])
    mu_d    = uniform(0.7, 1.5, u[13])
    sigma_d = jeffreys(0.05, 2., u[14])
    xi_d    = uniform(-10., 10., u[15])

    return a, mu_a, sigma_a, xi_a, \
           b, mu_b, sigma_b, xi_b, \
           g, mu_g, sigma_g, xi_g, \
           d, mu_d, sigma_d, xi_d


#for VF
print("VoidFinder (VF)")
plot_hist(data1_wall_vf,data2_void_vf,bins_,label_vf)
#model 1
print("Running Model 1 using VF...")

Model_1_fit(bounds1,data1_wall_vf,data2_void_vf,bins_,label_vf)
Model_1_sampler(prior_xform1,data1_wall_vf,data2_void_vf,bins_,label_vf)
lnZ1_vf = Model1_output(data1_wall_vf,data2_void_vf,bins_,label_vf)

#model2
print("Running Model 2 using VF...")

Model_2_fit(bounds2,data1_wall_vf,data2_void_vf,bins_,label_vf)
Model_2_sampler(prior_xform2,data1_wall_vf,data2_void_vf,bins_,label_vf)
lnZ2_vf = Model2_output(data1_wall_vf,data2_void_vf,bins_,label_vf)


#Baye's Factor
lnB12_vf = lnZ1_vf - lnZ2_vf
logB12_vf = 0.434 * lnB12_vf

print("Log of Baye's Factor for g-r using VF is: ", logB12_vf)


#for V2
print("V2")
plot_hist(data1_wall_v2,data2_void_v2,bins_,label_v2)
print("Running Model 1 using V2...")

#model 1
Model_1_fit(bounds1,data1_wall_v2,data2_void_v2,bins_,label_v2)
Model_1_sampler(prior_xform1,data1_wall_v2,data2_void_v2,bins_,label_v2)
lnZ1_v2 = Model1_output(data1_wall_v2,data2_void_v2,bins_,label_v2)

print("Running Model 2 using V2...")

#model2
Model_2_fit(bounds2,data1_wall_v2,data2_void_v2,bins_,label_v2)
Model_2_sampler(prior_xform2,data1_wall_v2,data2_void_v2,bins_,label_v2)
lnZ2_v2 = Model2_output(data1_wall_v2,data2_void_v2,bins_,label_v2)


#Baye's Factor
lnB12_v2 = lnZ1_v2 - lnZ2_v2
logB12_v2 = 0.434 * lnB12_v2

print("Log of Baye's Factor for g-r using V2 is: ", logB12_v2)
