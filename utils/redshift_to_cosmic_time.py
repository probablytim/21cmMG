from scipy.integrate import quad
import numpy as np

Ho = 0.6766*3.2407e-18
OMm = 0.31
OMl = 1 - OMm
OMr = 8.6e-5

def hubble(z):
    return Ho * (OMm * pow(1+z, 3) + OMr * pow(1+z, 4) + OMl)**(1/2);

def integrand(z):
    return 1 / (hubble(z) * (1+z))

def cosmic_time(z_to=0, z_from=np.inf):
    I =  quad(integrand, z_to, z_from)
    time = (I[0]) / (31556952000000) # in million years

    start = "big bang" if z_from == np.inf else "z=" + str(z_from)
    print(f"{time:.5f} million years from " + start + " until z=" + str(z_to))

cosmic_time(0)