import tempfile
import unittest
from pathlib import Path

import numpy as np

from ctwoodgrad import (
    fibre_sigmas_for_spacing,
    getFibreTensor,
    getFibreTensorForVoxelSize,
    staged_fibre_tensor_chunks,
)


class FibreChunkTests(unittest.TestCase):
    def test_spacing_calibration_matches_existing_pipeline(self):
        self.assertEqual(fibre_sigmas_for_spacing(0.5), (1.2, 2.5))

    def test_chunked_fields_match_full_volume_on_two_axes(self):
        scan = np.random.default_rng(7).normal(size=(57, 20, 23)).astype(np.float32)
        for axis, data in ((0, scan), (2, np.moveaxis(scan, 0, 2))):
            native = getFibreTensorForVoxelSize(data, 0.5)
            explicit = getFibreTensor(data, sigma=1.2, omega=2.5)
            for a, b in zip(native, explicit):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

            reconstructed = [np.empty((*data.shape, 3), dtype=np.float32) for _ in range(3)]
            with tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary)
                with staged_fibre_tensor_chunks(data, sigma=1.2, omega=2.5,
                                                axis=axis, chunk_slices=19,
                                                workers=2, memory_budget_gb=1,
                                                cache_dir=cache) as chunks:
                    intervals = []
                    for chunk in chunks:
                        intervals.append((chunk.start, chunk.stop))
                        target = [slice(None)] * 4
                        target[axis] = slice(chunk.start, chunk.stop)
                        for output, field in zip(reconstructed, (chunk.radial,
                                                                 chunk.tangential,
                                                                 chunk.longitudinal)):
                            output[tuple(target)] = field
                self.assertEqual(intervals, [(0, 19), (19, 38), (38, 57)])
                self.assertFalse(list(cache.iterdir()))
            for output, reference in zip(reconstructed, native):
                np.testing.assert_allclose(output, np.asarray(reference),
                                           atol=2e-5 if axis == 0 else 1e-4)

    def test_chunk_size_adapts_to_budget(self):
        scan = np.random.default_rng(3).normal(size=(60, 8, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            with staged_fibre_tensor_chunks(scan, sigma=0.7, omega=1.5,
                                            chunk_slices=20,
                                            memory_budget_gb=.0003,
                                            cache_dir=temporary) as chunks:
                intervals = [(chunk.start, chunk.stop) for chunk in chunks]
        self.assertGreater(len(intervals), 1)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], len(scan))


if __name__ == "__main__":
    unittest.main()
