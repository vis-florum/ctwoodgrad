import numpy as np

from ctwoodgrad import segmentation


def test_threshold_slicewise_uses_requested_slice_axis(monkeypatch):
    img_axis0 = np.stack(
        [
            np.full((3, 4), 10),
            np.full((3, 4), 20),
            np.full((3, 4), 30),
        ],
        axis=0,
    )
    img_axis2 = np.moveaxis(img_axis0, 0, 2)

    def threshold_from_slice_value(slice_np, lower, latewood):
        return slice_np[0, 0]

    monkeypatch.setattr(segmentation, "findInterMode", threshold_from_slice_value)

    mask_axis0, ts_axis0 = segmentation.threshold_slicewise_MT(
        img_axis0,
        lower=0,
        intermode_limit=100,
        upper=100,
        max_workers=2,
        slice_axis=0,
    )
    mask_axis2, ts_axis2 = segmentation.threshold_slicewise_MT(
        img_axis2,
        lower=0,
        intermode_limit=100,
        upper=100,
        max_workers=2,
        slice_axis=2,
    )

    np.testing.assert_array_equal(ts_axis0, np.array([10, 20, 30]))
    np.testing.assert_array_equal(ts_axis2, ts_axis0)
    assert mask_axis0.shape == img_axis0.shape
    assert mask_axis2.shape == img_axis2.shape
    np.testing.assert_array_equal(np.moveaxis(mask_axis2, 2, 0), mask_axis0)


def test_fill_cavities_slicewise_restores_requested_axis(monkeypatch):
    mask_axis0 = np.zeros((2, 3, 4), dtype=bool)
    mask_axis0[1] = True
    mask_axis2 = np.moveaxis(mask_axis0, 0, 2)

    monkeypatch.setattr(segmentation, "fillCavities", lambda mask: ~mask)

    filled_axis0 = segmentation.fill_cavities_slicewise_serial(mask_axis0, slice_axis=0)
    filled_axis2 = segmentation.fill_cavities_slicewise_serial(mask_axis2, slice_axis=2)

    assert filled_axis0.shape == mask_axis0.shape
    assert filled_axis2.shape == mask_axis2.shape
    np.testing.assert_array_equal(filled_axis0, ~mask_axis0)
    np.testing.assert_array_equal(np.moveaxis(filled_axis2, 2, 0), filled_axis0)
