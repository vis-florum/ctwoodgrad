import numpy as np
import os
import diplib as dip
import nrrd
from ctwoodgrad.visualisations import exportToVTK
# from ctwoodgrad import 

file = "/media/Store-HDD/johannes-data/CT-Data/RAW/1_cropped/L/L015.nrrd"

image, header = nrrd.read(file)
voxel_size_mm = header["spacings"]

mask_wood = image > 150

def contrast_stretch_wood(
    image,
    lower_bound: float,
    upper_bound: float,
) -> dip.Image:
    """Clip wood CT intensities and scale them to ``[0, 1]``."""
    if not lower_bound < upper_bound:
        raise ValueError("lower_bound must be smaller than upper_bound")
    image_dip = dip.Image(np.asarray(image, dtype=float))
    clipped = dip.Clip(image_dip, low=lower_bound, high=upper_bound)
    return (clipped - lower_bound) / (upper_bound - lower_bound)


def prepare_hessian(
    image,
    mask_wood,
    voxel_size_mm: tuple[float, float, float],
    *,
    reference_spacing_mm: float = 0.3,
    gradient_sigma_at_reference_px: float = 0.7,
    tensor_sigma_at_reference_px: float = 1.5,
    lower_bound: float = 150,
    upper_bound: float = 2000,
    ):
    
    array = np.asarray(image)
    wood = np.asarray(mask_wood, dtype=bool)
    spacing = np.asarray(voxel_size_mm, dtype=float)
    normalized = contrast_stretch_wood(array, lower_bound=lower_bound, upper_bound=upper_bound)
    mask_wood = dip.Image(wood)

    # Preserve the empirical scaling from the supplied DIPLib prototype. DIPLib
    # uses x/y order, whereas NumPy and voxel_size_mm use row/column order.
    mean_spacing = spacing[0]
    scale = mean_spacing / reference_spacing_mm
    sigma_mm = max(0.1, gradient_sigma_at_reference_px * scale) * mean_spacing
    omega_mm = max(0.1, tensor_sigma_at_reference_px * scale) * mean_spacing
    # DIPLib parameters are x/y ordered; make the empirical isotropic setting
    # physically isotropic when NumPy row/column spacing differs.
    sigma = 3*[round(sigma_mm / spacing[1], 2)]
    omega = 3*[round(omega_mm / spacing[1], 2)]

    hessian = dip.Hessian(normalized, sigmas=sigma)
    eigenvalues, _ = dip.EigenDecomposition(hessian)
    hessian_trace = dip.Trace(eigenvalues)
    # negative = hessian_trace < 0
    # negative_count = int(np.count_nonzero(np.asarray(negative)))
    # if negative_count:
    #     hessian_threshold = dip.Percentile(hessian_trace[negative], dip.Image(), 50)[0]
    #     earlywood_boundary = hessian_trace < hessian_threshold
    # else:
    #     earlywood_boundary = dip.Image(np.zeros(array.shape, dtype=bool))

    return hessian_trace


def gst_orientations(
    image,
    mask_wood,
    voxel_size_mm: tuple[float, float, float],
    *,
    reference_spacing_mm: float = 0.3,
    gradient_sigma_at_reference_px: float = 0.7,
    tensor_sigma_at_reference_px: float = 1.5,
    lower_bound: float = 150,
    upper_bound: float = 2000,
    ):
    
    array = np.asarray(image)
    wood = np.asarray(mask_wood, dtype=bool)
    spacing = np.asarray(voxel_size_mm, dtype=float)
    normalized = contrast_stretch_wood(array, lower_bound=lower_bound, upper_bound=upper_bound)
    mask_wood = dip.Image(wood)

    # Preserve the empirical scaling from the supplied DIPLib prototype. DIPLib
    # uses x/y order, whereas NumPy and voxel_size_mm use row/column order.
    mean_spacing = spacing[0]
    scale = mean_spacing / reference_spacing_mm
    sigma_mm = max(0.1, gradient_sigma_at_reference_px * scale) * mean_spacing
    omega_mm = max(0.1, tensor_sigma_at_reference_px * scale) * mean_spacing
    # DIPLib parameters are x/y ordered; make the empirical isotropic setting
    # physically isotropic when NumPy row/column spacing differs.
    sigma = 3*[round(sigma_mm / spacing[1], 2)]
    omega = 3*[round(omega_mm / spacing[1], 2)]

    g = dip.Gradient(normalized,sigmas=sigma)
    S = g @ dip.Transpose(g)
    dip.Gauss(S, out=S, sigmas=omega)
    eigenvalues, eigenvectors = dip.EigenDecomposition(S)
    v1 = eigenvectors.TensorColumn(0)
    v2 = eigenvectors.TensorColumn(1)
    v3 = eigenvectors.TensorColumn(2)
    # energy, phi1, theta1 = dip.StructureTensorAnalysis(S,outputs=["energy", "phi1", "theta1"])
    
    return v1, v2, v3, # energy, phi1, theta1

hessian_trace = prepare_hessian(image,mask_wood,voxel_size_mm,
                                gradient_sigma_at_reference_px=5)


v1, v2, v3 = gst_orientations(image,mask_wood,voxel_size_mm,
                                # gradient_sigma_at_reference_px=2.5,
                                # tensor_sigma_at_reference_px=5
                                )

outarray = np.asarray(hessian_trace)
header_new = header.copy()
header_new["type"] = "Float64"

outfile = "/home/aime/Desktop/L015-hessian.nrrd"
nrrd.write(outfile,outarray,header_new)


fieldnames = ["v11", "v12", "v13"]

v1_1, v1_2, v1_3 = v1.TensorRow(0), v1.TensorRow(1), v1.TensorRow(2)
# Lieber die WInkel vom GST!
for i,field in enumerate([v3_1, v3_2, v3_3]):
    outarray = np.asarray(field)
    header_new = header.copy()
    header_new["type"] = "Float64"

    outfile = "/home/aime/Desktop/L015-gst-" + fieldnames[i] + ".nrrd"
    nrrd.write(outfile,outarray,header_new)

outfile = "/home/aime/Desktop/L015-gst-v3.vti"
exportToVTK(outfile, v3=v3)
