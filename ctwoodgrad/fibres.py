import diplib as dip
import numpy as np
import logging

from .geometry import getSampleAxis

def getFCS(img, sigma=.7, omega=1.5):
#def getFCS(img, sigma=3, omega=5.0):    # wide annual rings oak
#def getFCS(img, sigma=1.5, omega=3.0):    # small annual rings oak
    '''Get the Fibre Coordinate System of the image'''
    # Normalisation
    imgn = img - np.min(img)
    imgn = imgn / np.max(imgn)

    g = dip.Gradient(imgn,sigmas=sigma)
    S = g @ dip.Transpose(g)
    dip.Gauss(S, out=S, sigmas=omega)
    eigenvalues, eigenvectors = dip.EigenDecomposition(S)
    v1 = eigenvectors.TensorColumn(0)
    v2 = eigenvectors.TensorColumn(1)
    v3 = eigenvectors.TensorColumn(2)
    #energy, cyl, plan = dip.StructureTensorAnalysis(S,outputs=["energy", "cylindrical", "planar"])

    return v1,v2,v3,# energy, cyl, plan


def getFibreTensor(img, sigma=.7, omega=1.5):
    '''Fibre estimations in the object'''
    try:
        R, T, L = getFCS(img, sigma=sigma, omega=omega)
        return R, T, L
    except Exception as e:
        logging.error(f"getFibreTensor failed: {e}")
        return None, None, None


def projectDipDir(dipL, ax):
    '''Project the L vectors (diplib tensor image) onto arbitrary axis
    https://github.com/DIPlib/diplib/blob/master/examples/python/04_tensor_images.ipynb'''
    dipProj = dip.Transpose(dipL) @ ax
    #dipProj = dip.DotProduct(dipL,dip.Create0D(ax))   # alternatively
    dip.Abs(dipProj, out=dipProj)

    return dipProj


def getFibreAlignment(v3,mask_wood):
    '''Get the fibre alignment with the longitudinal axis of the wood mask'''    
    I_l = getSampleAxis(mask_wood)
    L_proj = projectDipDir(v3, I_l)
    L_proj_w = L_proj[mask_wood]
    L_prop = dip.Sum(L_proj_w)[0][0] / len(L_proj_w)   # proportion of L along loading axis, only in wood piece

    return L_prop



# #################################################################################################################

# # TODO:
# # Statistics about Orientations!

# # Sum of z component of  L vectors
# L = np.array(v3)
# L_w = L[mask_wood,:]

# L_w_z = np.abs(L[mask_wood,0])
# L_w_m = np.mean(L_w)
# plt.hist(L_w, bins=100)
# plt.show()

# # from the vector field calculate the inclination and azimuth for each vector
# # inclination = angle between vector and z axis
# # azimuth = angle between projection of vector on xy plane and x axis
# # inclination = arccos(z / |v|)
# # azimuth = arctan(y / x)
# ori = dip.Orientation(v3)
# ori = np.array(ori)
# ori_w = ori[mask_wood,:]


# L_w_z = L_w[:,0]
# L_w_y = L_w[:,1]
# L_w_x = L_w[:,2]
# r = np.hypot(L_w_x, L_w_y)


# inclination = np.arccos(L_w_z)
# #azimuth = np.arccos(L_x, L_y)

# incl_shift = np.where(inclination > np.pi/2 , inclination - np.pi, inclination)

# # have angle mapped to 0, 2pi
# inclination = np.mod(inclination, np.pi)

# plt.hist(inclination, bins=100)
# plt.hist(incl_shift, bins=200)
# #plt.hist(azimuth, bins=100)
# plt.hist(L_phi_w, bins=200)
# plt.hist(L_th_w, bins=200)
# plt.show()


# i = 222
# a=L_w[i]
# ori_w[i]
# L_phi_w[i]
# L_th_w[i]
# inclination[i]

# x = a[0]
# y = a[1]
# z = a[2]
# np.arctan(z/np.sqrt(x**2 + y**2))
# np.arctan2(z,np.sqrt(x**2 + y**2))-np.pi/2

# imgn = img - np.min(img)
# imgn = imgn / np.max(imgn)

# GST = dip.StructureTensor(imgn, gradientSigmas=sigma, tensorSigmas=omega)
# L_phi, L_th = dip.StructureTensorAnalysis(GST,outputs=["phi3", "theta3"])

# L_phi = np.array(L_phi)
# L_th = np.array(L_th)

# L_phi_w = L_phi[mask_wood]
# L_th_w = L_th[mask_wood]

# L_phi_m = np.mean(L_phi)
# L_th_m = np.mean(L_th)

# plt.hist(L_phi, bins=200)
# plt.show()
# plt.hist(L_th, bins=200)
# plt.show()

# # TODO:
# # 2D histogram of L_phi and L_th
# plt.hist2d(L_phi, L_th, bins=300)
# plt.show()
    
