import diplib as dip
import numpy as np
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

    if dip.AreDimensionsReversed():
        logging.debug("diplib.ReverseDimensions()")
        dip.ReverseDimensions()

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

    if dip.AreDimensionsReversed():
        dip.ReverseDimensions()
    
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

# Needs: from pyevtk.hl import imageToVTK
# requires pyevtk

def createVTK(outfile, rho, mask_wood, labels_ewlw, dipR, dipT, dipL, slices=None):
    '''Create a VTK file from the image and the FCS'''
    from pyevtk.hl import imageToVTK
    
    if slices is not None:
        rho = rho[slices[0]:slices[1],:,:]
        mask_wood = mask_wood[slices[0]:slices[1],:,:]
        labels_ewlw = labels_ewlw[slices[0]:slices[1],:,:]
        dipR = dipR[slices[0]:slices[1],:,:]
        dipT = dipT[slices[0]:slices[1],:,:]
        dipL = dipL[slices[0]:slices[1],:,:]
        

    def prepareDirsVTK(v):
        v = np.array(v)
        v = np.asfortranarray(v) # Need fortran order for vectors in imageToVTK
        if dip.AreDimensionsReversed():
            vx = v[:,:,:,2]
            vy = v[:,:,:,1]
            vz = v[:,:,:,0]
        else:
            vx = v[:,:,:,0]
            vy = v[:,:,:,1]
            vz = v[:,:,:,2]
        
        return (vx, vy, vz)

    rdir = prepareDirsVTK(dipR)
    tdir = prepareDirsVTK(dipT)
    ldir = prepareDirsVTK(dipL)
    woodmask = mask_wood.astype("uint16")

    imageToVTK(outfile, pointData = {"density" : rho, "wood" : woodmask, "EW-LW" : labels_ewlw,
                                     "Rdir" : rdir, "Tdir" : tdir, "Ldir" : ldir})
    

def fibreToVTK(outfile, dipR, dipT, dipL):
    '''Create a VTK file from FCS'''
    from pyevtk.hl import imageToVTK

    def prepareDirsVTK(v):
        v = np.array(v)
        v = np.asfortranarray(v) # Need fortran order for vectors in imageToVTK
        if dip.AreDimensionsReversed():
            vx = v[:,:,:,2]
            vy = v[:,:,:,1]
            vz = v[:,:,:,0]
        else:
            vx = v[:,:,:,0]
            vy = v[:,:,:,1]
            vz = v[:,:,:,2]
        
        return (vx, vy, vz)

    rdir = prepareDirsVTK(dipR)
    tdir = prepareDirsVTK(dipT)
    ldir = prepareDirsVTK(dipL)

    imageToVTK(outfile, pointData = {"Rdir" : rdir, "Tdir" : tdir, "Ldir" : ldir})
    
    
def fibreRhoToVTK(outfile, rho, dipR, dipT, dipL):
    '''Create a VTK file from FCS'''
    from pyevtk.hl import imageToVTK

    def prepareDirsVTK(v):
        v = np.array(v)
        v = np.asfortranarray(v) # Need fortran order for vectors in imageToVTK
        if dip.AreDimensionsReversed():
            vx = v[:,:,:,2]
            vy = v[:,:,:,1]
            vz = v[:,:,:,0]
        else:
            vx = v[:,:,:,0]
            vy = v[:,:,:,1]
            vz = v[:,:,:,2]
        
        return (vx, vy, vz)

    rdir = prepareDirsVTK(dipR)
    tdir = prepareDirsVTK(dipT)
    ldir = prepareDirsVTK(dipL)

    imageToVTK(outfile, pointData = {"density" : rho, "Rdir" : rdir, "Tdir" : tdir, "Ldir" : ldir})
    




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









# #################################################################################################################
# # Visualisations



# sigma = .7
# omega = 1.5
# v1, v2, v3, energy, cyl, plan = getFCS(imgn, sigma, omega)
# v1, v2, v3, energy, cyl, plan = getFCSinv(img, sigma, omega)

# GST = dip.StructureTensor(imgn, gradientSigmas=sigma, tensorSigmas=omega)
# L_phi, L_th = dip.StructureTensorAnalysis(GST,outputs=["phi3", "theta3"])

# energy = np.array(energy)
# cyl = np.array(cyl)
# plan = np.array(plan)
# phi = np.array(L_phi)
# theta = np.array(L_th)

# def getOrientationData(v):
#     v = np.array(v)
#     v = np.asfortranarray(v) # Need fortran order for vectors in imageToVTK
#     if dip.AreDimensionsReversed():
#         vx = v[:,:,:,2]
#         vy = v[:,:,:,1]
#         vz = v[:,:,:,0]
#     else:
#         vx = v[:,:,:,0]
#         vy = v[:,:,:,1]
#         vz = v[:,:,:,2]
    
#     return (vx, vy, vz)

# R = getOrientationData(v1)
# T = getOrientationData(v2)
# L = getOrientationData(v3)


# outfile = os.path.join(out_dir, id + "_orientations")
# # outfile = os.path.join(out_dir, "ellipsoid_orients")
# imageToVTK(outfile, pointData = {"density" : density_data, "Rdir" : R, "Tdir" : T, "Ldir" : L,
#                                  "energy" : energy, "cyl" : cyl, "plan" : plan})

# outfile = os.path.join(out_dir, id + "_orientations_inv")
# imageToVTK(outfile, pointData = {"density" : img, "Rdir" : R, "Tdir" : T, "Ldir" : L,
#                                  "energy" : energy, "cyl" : cyl, "plan" : plan,
#                                  "phi" : phi, "theta" : theta})



# # Hessian
# H = dip.Hessian(imgn)
# H.Show()

# # Hessian energy (sum of eigenvalues)
# eigenvalues, eigenvectors = dip.EigenDecomposition(H)
# E = dip.Trace(eigenvalues)
# E.Show()

# h1 = eigenvectors.TensorColumn(0)
# h2 = eigenvectors.TensorColumn(1)
# h3 = eigenvectors.TensorColumn(2)

# E = np.array(E)

# imageToVTK(outfile, pointData = {"density" : density_data, "Rdir" : R, "Tdir" : T, "Ldir" : L,
#                                  "HessE" : E,
#                                  "energy" : energy, "cyl" : cyl, "plan" : plan})


