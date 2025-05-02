import diplib as dip
import numpy as np
from pyevtk.hl import imageToVTK
import logging


def getSampleAxis(mask):
    '''Get the principal axis of the binary sample mask (longest axis of the sample))'''
    #A = dip.Image(img)
    dipimg = dip.Image()
    dipB = dip.Label(dip.Image(mask), minSize=1000)
    m = dip.MeasurementTool.Measure(dipB, dipimg, ["MajorAxes"])
    I_l = m["MajorAxes"][1][-3:]    # inertia axis for length of sample has lowes EV

    return I_l


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


def findEWLW(img, mask_wood):
    '''Find the EW / LW content of the wood mask using curvature analysis'''
    imgn = img - np.min(img)
    imgn = imgn / np.max(imgn)
    
    logging.debug("Building Hessian... ")
    H = dip.Hessian(imgn)
    logging.debug(f"    {H}")

    logging.debug("Eigen decomposition... ")
    eigenvalues, eigenvectors = dip.EigenDecomposition(H)
    logging.debug(f"    {eigenvalues}... ")
    logging.debug(f"    {eigenvectors}... ")

    # Hessian Energy
    logging.debug("Get Laplacian... ")
    H_E = dip.Trace(eigenvalues)
    H_E = np.array(H_E)

    # EW / LW content
    logging.debug("Masking... ")
    mask_ew = mask_wood & (H_E < 0)
    mask_lw = mask_wood & (H_E >= 0)
    
    # Ratios
    logging.debug("EW/LW rations... ")
    r_ew = np.sum(mask_ew) / np.sum(mask_wood)
    r_lw = np.sum(mask_lw) / np.sum(mask_wood)
    # check if r_ew + r_lw is approximately 1
    if abs(r_ew + r_lw - 1) > 0.01:
        logging.error("Warning: r_ew + r_lw = " + str(r_ew + r_lw) + ", instead of 1!")

    # Create a labael map where 0=background, 1=EW, 2=LW
    logging.debug("Creating label map... ")
    labels_ewlw = np.zeros_like(img)
    labels_ewlw[mask_ew] = 1
    labels_ewlw[mask_lw] = 2
    
    return labels_ewlw, r_ew



################# VISUALISATIONS

def prepareDirsVTK(dip_vecs):
    """Convert a DIPlib vector image into a (fx, fy, fz) Fortran-style tuple."""
    if dip.AreDimensionsReversed(): # this is standard order in diplib
        fx = np.asfortranarray(dip_vecs(2))
        fy = np.asfortranarray(dip_vecs(1))
        fz = np.asfortranarray(dip_vecs(0))
    else:
        fx = np.asfortranarray(dip_vecs(0))
        fy = np.asfortranarray(dip_vecs(1))
        fz = np.asfortranarray(dip_vecs(2))
    
    return (fx, fy, fz)


def processField(value):
    """Prepare data field based on whether it's a DIPlib image or NumPy array."""
    if isinstance(value, dip.Image):
        if value.TensorElements() == 3:
            # It's a vector field
            return prepareDirsVTK(value)
        elif value.TensorElements() == 1:
            # Scalar image
            return np.asfortranarray(value)
        else:
            raise ValueError(f"Unsupported number of tensor elements for VTK export: {value.TensorElements()}")
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return value.astype(np.uint16)
        else:
            return value
    else:
        raise TypeError(f"Unsupported type for VTK export: {type(value)}")
    

def exportToVTK(outfile, **kwargs):
    """
    General-purpose VTK exporter. Automatically handles dip.Image and numpy arrays.
    Converts:
      - DIPlib vector images to VTK vector fields
      - DIPlib scalar images to Fortran-style arrays
      - int numpy arrays to uint16
      - other numpy arrays left as-is
    """
    pointData = {}
    for key, val in kwargs.items():
        pointData[key] = processField(val)

    imageToVTK(outfile, pointData=pointData)
    return





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
