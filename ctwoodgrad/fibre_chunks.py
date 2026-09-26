"""Disk-backed, lengthwise fibre tensors for memory-bounded CT processing."""

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import math
import multiprocessing
import os
from operator import index
from pathlib import Path
import tempfile

import diplib as dip
import numpy as np

from .fibres import _fibre_tensor_from_normalized, fibre_sigmas_for_spacing


@dataclass(frozen=True)
class FibreTensorChunk:
    """A non-overlapping core of the three direction fields."""

    start: int
    stop: int
    axis: int
    radial: np.ndarray
    tangential: np.ndarray
    longitudinal: np.ndarray


@dataclass(frozen=True)
class _ChunkPlan:
    start: int
    stop: int
    read_start: int
    read_stop: int


def _chunk_plans(length, core_slices, halo):
    return [
        _ChunkPlan(start, min(start + core_slices, length),
                   max(0, start - halo), min(length, start + core_slices + halo))
        for start in range(0, length, core_slices)
    ]


def _calculate_chunk(density, chunk, sigma, omega, norm_min, norm_max,
                     destination, threads, dimensions_reversed, axis):
    # Spawned processes begin with DIPlib's default dimension convention.
    if dip.AreDimensionsReversed() != dimensions_reversed:
        dip.ReverseDimensions()
    previous_threads = dip.GetNumberOfThreads()
    dip.SetNumberOfThreads(threads)
    try:
        selection = [slice(None)] * 3
        selection[axis] = slice(chunk.read_start, chunk.read_stop)
        slab = np.ascontiguousarray(density[tuple(selection)])
        normalized = (slab - norm_min) / norm_max
        directions = _fibre_tensor_from_normalized(normalized, sigma, omega)
        local = [slice(None)] * 4
        local[axis] = slice(chunk.start - chunk.read_start,
                            chunk.stop - chunk.read_start)
        core_shape = list(density.shape)
        core_shape[axis] = chunk.stop - chunk.start
        for name, direction in zip(("radial", "tangential", "longitudinal"), directions):
            target = destination / f"{name}_{chunk.start:08d}.npy"
            array = np.lib.format.open_memmap(
                target, mode="w+", dtype=np.float32,
                shape=(*core_shape, 3),
            )
            array[:] = np.asarray(direction)[tuple(local)]
            array.flush()
            del array
    finally:
        dip.SetNumberOfThreads(previous_threads)
    return chunk


def _calculate_from_file(args):
    source, chunk, sigma, omega, norm_min, norm_max, destination, threads, dimensions_reversed, axis = args
    density = np.load(source, mmap_mode="r")
    return _calculate_chunk(density, chunk, sigma, omega, norm_min, norm_max,
                            destination, threads, dimensions_reversed, axis)


def _stage_chunks(density, sigma, omega, destination, axis, core_slices, workers,
                  memory_budget_gb):
    halo = math.ceil(3 * sigma) + math.ceil(3 * omega) + 1
    length = density.shape[axis]
    bytes_per_slice = density.size // length * 160
    budget_bytes = memory_budget_gb * 1024**3
    if min(length, core_slices + 2 * halo) * bytes_per_slice > budget_bytes:
        allowed_core = int(budget_bytes // bytes_per_slice) - 2 * halo
        if allowed_core < 1:
            raise ValueError("The fibre memory budget cannot hold a slice plus both "
                             "Gaussian halos; increase memory_budget_gb.")
        logging.info("Reducing fibre chunk size from %d to %d for the memory budget",
                     core_slices, allowed_core)
        core_slices = allowed_core
    chunks = _chunk_plans(length, core_slices, halo)

    # Reproduce the full-volume normalization, including integer arithmetic.
    norm_min = density.min()
    norm_max = max(
        np.max(density[tuple(slice(start, start + core_slices) if i == axis
                             else slice(None) for i in range(3))] - norm_min)
        for start in range(0, length, core_slices)
    )
    if norm_max == 0:
        raise ValueError("Cannot calculate fibre directions for a constant-density volume")

    largest = max(chunk.read_stop - chunk.read_start for chunk in chunks)
    estimated_bytes = largest * bytes_per_slice
    workers = min(workers, len(chunks))
    threads = max(1, min(8, (os.cpu_count() or 1) // workers))
    dimensions_reversed = dip.AreDimensionsReversed()
    logging.info("Fibre chunks: %d, core=%d, halo=%d, workers=%d, "
                 "estimated peak per worker=%.2f GiB (budget %.2f GiB each)",
                 len(chunks), core_slices, halo, workers,
                 estimated_bytes / 1024**3, memory_budget_gb)

    if workers == 1:
        for chunk in chunks:
            _calculate_chunk(density, chunk, sigma, omega, norm_min, norm_max,
                             destination, threads, dimensions_reversed, axis)
            logging.info("Staged fibre slices %d:%d", chunk.start, chunk.stop)
    else:
        source = destination / "density.npy"
        mapped = np.lib.format.open_memmap(source, mode="w+", dtype=density.dtype,
                                           shape=density.shape)
        mapped[:] = density
        mapped.flush()
        del mapped
        tasks = ((source, chunk, sigma, omega, norm_min, norm_max, destination,
                  threads, dimensions_reversed, axis) for chunk in chunks)
        # DIPlib owns a native thread pool; fork can hang after it has run.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for chunk in pool.map(_calculate_from_file, tasks):
                logging.info("Staged fibre slices %d:%d", chunk.start, chunk.stop)
        source.unlink()
    return chunks


def _iter_staged_chunks(chunks, destination, axis):
    for chunk in chunks:
        paths = [destination / f"{name}_{chunk.start:08d}.npy"
                 for name in ("radial", "tangential", "longitudinal")]
        arrays = [np.load(path, mmap_mode="r+") for path in paths]
        result = FibreTensorChunk(chunk.start, chunk.stop, axis, *arrays)
        yield result
        del result, arrays
        for path in paths:
            path.unlink()
        logging.info("Released fibre slices %d:%d", chunk.start, chunk.stop)


@contextmanager
def staged_fibre_tensor_chunks(density, voxel_spacing_mm, *, axis=0,
                               chunk_slices=256, workers=1, memory_budget_gb=6.0,
                               cache_dir=None):
    """Yield disk-backed direction chunks with global normalization.

    The output arrays have the input's spatial axis order plus a final vector
    dimension. Only each core is yielded; Gaussian halos are discarded.
    The temporary files are removed on context exit, including after errors.
    """
    density = np.asarray(density)
    if density.ndim != 3:
        raise ValueError("density must have three dimensions")
    axis = index(axis)
    if axis < 0:
        axis += density.ndim
    if axis < 0 or axis >= density.ndim:
        raise ValueError("axis must identify one of the three density dimensions")
    if chunk_slices < 1 or workers < 1 or memory_budget_gb <= 0:
        raise ValueError("chunk_slices, workers, and memory_budget_gb must be positive")
    sigma, omega = fibre_sigmas_for_spacing(voxel_spacing_mm)
    cache_parent = Path(cache_dir) if cache_dir is not None else None
    if cache_parent is not None:
        cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ctwoodgrad-fibres-", dir=cache_parent) as temporary:
        destination = Path(temporary)
        plans = _stage_chunks(density, sigma, omega, destination, axis,
                              chunk_slices, workers, memory_budget_gb)
        yield _iter_staged_chunks(plans, destination, axis)
