import numpy as np
import os
import pytest
import diplib as dip
from ctwoodgrad import findInterMode, segmentAir, getFCS, exportToVTK

def test_findInterMode():
    img = np.random.randint(0, 2000, (70, 80, 100))
    t = findInterMode(img, 500, 1500)
    assert 500 <= t <= 1500, "Threshold should be within given bounds"

def test_segmentAir():
    img = np.random.randint(0, 2000, (50, 50, 50))
    mask, _ = segmentAir(img)
    assert mask.shape == img.shape
    
def create_fuzzy_ellipsoid(shape=(42, 64, 128), center=None, radii=(15, 25, 40), transition=5):
    """Create a 3D numpy array with a fuzzy ellipsoid: core is 1, fading to 0."""
    if center is None:
        center = [s // 2 for s in shape]
    z, y, x = np.indices(shape)
    normed = (((z - center[0]) / radii[0]) ** 2 +
              ((y - center[1]) / radii[1]) ** 2 +
              ((x - center[2]) / radii[2]) ** 2)
    smooth = np.clip(1.0 - (normed - 1) / (transition / min(radii)), 0.0, 1.0)
    return smooth.astype(np.float32)

def test_export_ellipsoid_with_fcs():
    # Parameters
    output_dir = "vtk_out"
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, "ellipsoid_fcs")

    # Create ellipsoid with soft edge
    ellipsoid = create_fuzzy_ellipsoid()

    # Run getFCS (assumed to return dipR, dipT, dipL as dip.Image objects with 3 tensor elements)
    dirR, dirT, dirL = getFCS(ellipsoid)

    # Export using general exporter
    exportToVTK(str(outfile), ellipsoid=ellipsoid, R=dirR, T=dirT, L=dirL)

    print(f"VTK output written to: {outfile}.vti or .vtk (depending on backend)")

    # Check that the VTK files exist
    for suffix in [".vti", ".vti"]:
        assert os.path.exists(str(outfile) + suffix) or os.path.exists(str(outfile) + suffix.replace('.vti', '.vtk'))

if __name__ == "__main__":
    pytest.main()
