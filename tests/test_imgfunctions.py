import numpy as np
from ctwoodgrad import findInterMode, segmentAir

def test_findInterMode():
    img = np.random.randint(0, 2000, (100, 100))
    t = findInterMode(img, 500, 1500)
    assert 500 <= t <= 1500, "Threshold should be within given bounds"

def test_segmentAir():
    img = np.random.randint(0, 2000, (50, 50, 50))
    mask, _ = segmentAir(img)
    assert mask.shape == img.shape

if __name__ == "__main__":
    pytest.main()
