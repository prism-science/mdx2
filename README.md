[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10519719.svg)](https://doi.org/10.5281/zenodo.10519719)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mdx2/badges/version.svg)](https://anaconda.org/conda-forge/mdx2)

# *mdx2*: macromolecular diffuse scattering data reduction in python

New here? Have a look at our recent [Blog post](https://prismscience.org/research/mdx2). See also [Examples](#examples) and [References](#references), below.

## Examples

- Introductory walkthrough using a small insulin dataset: [examples/insulin-tutorial](examples/insulin-tutorial/README.md)
- Scripts and notebooks to regenerate the figures from Meisburger & Ando, Acta Cryst. D (2024): [examples/insulin-multi-crystal](examples/insulin-multi-crystal).
- External repo that batch reprocesses Mac1 datasets for an upcoming article: [github.com/prism-science/diffuse-processing-mac1-east-west](https://github.com/prism-science/diffuse-processing-mac1-east-west)

## Versions

### Next version (unreleased, in `dev`)

Faster `mdx2.scale`: the same alternating-least-squares algorithm, with compiled multi-threaded passes over the data.

- `mdx2.scale` is 20-30x faster on large datasets and needs about half the memory. Two Eiger 16M sweeps (112M observations, `--mca2020`): 30 min, 40 GB -> 1 min, 17 GB on a 64-core node. Results match the previous version (chi2 at every refinement step; a few borderline outliers can differ).
- `--nproc` sets the number of threads; the default (-1) uses all CPUs available to the process (respecting the affinity mask and container CPU quotas).
- `--engine reference` runs the previous implementation (single-threaded; kept for one release).
- `--profile` logs wall time and memory per step; `--profile_file` writes them as JSON lines.
- Removed: pre-scaling (`--prescale.*`).
- `mdx2.merge` uses the same compiled multi-threaded passes (applying scales, merging, outliers, splits); about 2x faster on the insulin example, where most of the remaining time is loading and saving. It also has `--nproc` (default -1, all available CPUs), `--profile` and `--profile_file`.
- `mdx2.merge --split weightedRandomHalf`: weights are now accumulated in double precision. Previous versions summed float32 weights over the whole dataset, which put ~4% of observations in the wrong half (insulin example); half-dataset maps (and CC1/2) will differ slightly from earlier versions. Merged intensities agree to float32 precision.
- Outlier rejection (`mdx2.scale`, `mdx2.merge --outlier`) is now two-sided: in each group of symmetry-equivalent observations, the one with the largest leave-one-out residual (against the weighted mean of the others) is masked if beyond the cutoff, repeated after each scaling stage (scale) or until none remain (merge; at most 100 rounds, typically under 15); both observations of a disagreeing pair are masked. Previous versions masked only observations below the merged value, so high outliers stayed in, and good observations next to a high outlier could be masked. On the insulin example this masks ~3x as many observations (1573 vs 545; 0.02% of the data). `--outlier_method legacy` restores the previous behaviour (kept for one release; always used by `--engine reference`).
- `mdx2.mask_peaks` writes a *dynamic* mask by default: the peak model, the reflection conditions and the count-threshold peak pixels the model does not cover (insulin example: 15k of 294k pixels). `mdx2.integrate` and `mdx2.reintegrate` rebuild the mask chunk by chunk from the Miller indices they compute anyway, so the full boolean array is never formed: on the insulin example `mask_peaks` takes 1.8 s instead of 79 s and 0.6 GB instead of 6.6 GB (8.9 MB -> 0.1 MB file); integration is ~15-20% slower (the per-pixel peak test). The masks are identical pixel for pixel. `--static` writes the previous full-array mask; masks from earlier versions still work.
- `mdx2.import_data` uses much less memory and is several times faster: each batch of frames is assembled as uint16 (values above 65533 kept exactly in a side list) and written chunk by chunk, so memory per process is ~1/4 of before; Eiger (NXmx) and miniCBF (Pilatus) frames are read without dxtbx (the first frame is checked against dxtbx; other formats, or any difference, use dxtbx); frames are decoded and datastore chunks compressed in `--nthreads` threads per process (default 4). Eiger 16M sweep (7200 frames) on 32 cores: 210 s -> 40 s with `--nproc 8` (19 GB); Pilatus 6M (2800 frames): 31 s -> 7 s (9 GB). New dependencies: `bitshuffle`, `numcodecs`; `env.yaml` pins HDF5 below 2.0 for now (virtual datasets written by HDF5 2.x cannot be read by 1.14).
- `mdx2.import_data` compresses the datastore with Blosc (LZ4, bitshuffle) instead of gzip: reading ~3x faster (Eiger 16M chunk, one thread, Xeon: 283 -> 92 ms; per-chunk reads were a third of integration time on Eiger data), writing ~4x faster, and slightly smaller. Blosc datastores have no HDF5 byte-shuffle filter (datastores written by earlier development versions have one, which nexusformat adds by default, and read at about half that speed). `--compression blosc-zstd` is ~20% smaller, `--compression gzip` gives the previous format. Reading Blosc data needs the `hdf5plugin` filters (a dependency of mdx2, dxtbx and nexusformat; other HDF5 tools need it too). Existing datastores keep working.
- Faster `mdx2.integrate` and `mdx2.reintegrate`: each chunk is binned in one compiled pass (h, k, l, the peak mask and the sums per voxel, without full-size intermediate arrays; identical to the previous per-chunk result), and each task handles one column of chunks along phi, which is one detector module with the default chunking. Insulin example, 8 processes: integrate 102 s -> 16 s, reintegrate 116 s -> 17 s; Eiger 16M chunks ~8x faster per chunk with ~12x less working memory. `--profile` logs time per step; `--engine reference` runs the previous code (kept for one release). At most one process per chunk column is used.
- `mdx2.integrate`: partial observations are grouped (`--max_spread`) within each detector module, so an observation never combines pixels from different modules (previously, slivers across the narrow gaps between module columns were merged). Insulin example: 0.56% more observations, the same totals per voxel.
- `mdx2.reintegrate` output is reproducible: chunk results are added to the grid in a fixed order, so serial and parallel runs give bit-identical sums (previously float32 rounding depended on which chunk finished first; differences ~1e-7 relative).
- Faster `mdx2.find_peaks` with less memory: one task per detector module with a compiled threshold, and the Miller-index grid loaded only after the search; `MillerIndex.interpolate` no longer copies the grid three times (this also helps `mdx2.mask_peaks`: Eiger 16M 27 s -> 19 s, 13.4 GB -> 4.2 GB). Eiger 16M sweep on 32 cores: 79 s -> 40 s, main process 13.3 GB -> 5.2 GB; identical peaks and peak model. `--engine reference` runs the previous implementation (kept for one release); `--profile` logs time per step.
- Large arrays in mdx2 files (hkl tables, peaks, Miller-index grids, correction maps, masks) are compressed with Blosc LZ4 + byte shuffle instead of nexusformat's default gzip + shuffle: writing ~20x and reading ~3-5x faster, files ~10% larger. Eiger 16M sweep: `mdx2.correct` saves in 2 s instead of 33 s; the Miller-index grid writes in 3 s instead of 63 s and loads in 3.7 s instead of 10 s (`mdx2.find_peaks`, `mask_peaks`, `integrate`). Existing files read as before.
- Faster `mdx2.correct` with less memory: row blocks in `--nthreads` threads (default 4), results written straight into the output columns (no copy of the table), and a faster `Symmetry.to_asu` (also used by scale, merge, reintegrate). Eiger 16M sweep (53.7 M rows): 55 s -> 18 s, 11.6 GB -> 5.8 GB; identical output. `--engine reference` runs the previous implementation (kept for one release); `--profile` logs time per step.
- Faster grouping of symmetry-equivalent observations (`HKLTable.unique`), used by `mdx2.scale` and `mdx2.merge`.
- Scale files are unchanged, so downstream steps (`mdx2.merge`, `mdx2.reintegrate`, reports) work as before.
- New dependencies: `numba`, `threadpoolctl`, `psutil`.
- `env.yaml` no longer pins Python to 3.10 (dxtbx releases after 3.24, needed for recent Eiger files, are built for Python >= 3.11).

### Version 1.0.5

A minor update to prepare the CLI for pipeline integration. Previous versions had to be invoked from the working directory. This version allows arbitrary directories to be set for all input and output files.

- Feature: CLI tools have a `--logfile` option to explicitly set the path for logging (retaining the current default: `mdx2.<command>.log`)
- Bugfix: `mdx2.import_data` creates valid virtual dataset paths when `data.nxs` was written outside the current working directory

### Version 1.0.4

- Added `mdx2.report` to automatically generate and execute ipython notebook reports from the command-line. Available templates: `visualization`, `scaling_model`, and `map_statistics`. For example outputs, see: [examples/insulin-multi-crystal/reports](examples/insulin-multi-crystal/reports) and [examples/insulin-tutorial/reports](examples/insulin-tutorial/reports).

### Version 1.0.3

- Performance boost for `mdx2.import_data` using parallel read and write. The `data.nxs` file contains a virtual dataset linking to neXus files in a subdirectory (`datastore/` by default).
- `mdx2.reintegrate` -- New command-line tool to create fine maps after scaling (single-sweep only: multi-crystal datasets not yet implemented)
- Optional pre-scaling in `mdx2.scale` to correct anisotropic background
- Improved handling of command-line arguments via `dataclass` attributes and `simple-parsing` package
- Updated examples

### Version 1.0.2

- Rudimentary Bragg peak integration, in development
- Support for non-reference space group settings
- Bug fixes, including:
  - Symmetry operators now rotate in the correct direction
  - Gracefully skip missing or masked data chunks

## Installation

### Prerequisites

You'll need package manager for conda environments. If you don't have one already, we recommend installing [miniforge](https://github.com/conda-forge/miniforge), which includes conda and mamba commands, and has the conda-forge channel as its default. If you use a different manager, you'll need to add the flag `-c conda-forge`.

### User install (miniforge)

Minimal install:

```bash
mamba create -n mdx2
mamba activate mdx2
mamba install mdx2
```

Recommended: add packages for Bragg processing, jupyter notebooks, and nexus visualization

```bash
mamba install jupyterlab dials xia2 nexpy
```

## Performance notes

`mdx2.scale` runs its passes over the data on multiple threads (numba). Typical memory: about 150 bytes per observation (17 GB for 112M observations).

- Threads: `--nproc N` (default: all available CPUs). On shared machines, choose fewer.
- Threading layer: mdx2 uses numba's `workqueue` layer. Set `NUMBA_THREADING_LAYER` to override; numba's OpenMP layer gave wrong results on some multi-core Linux nodes, and mdx2 stops with an error if a parallel pass miscounts observations.
- Transparent huge pages: on Linux machines where huge pages are allocated with direct compaction (`/sys/kernel/mm/transparent_hugepage/defrag` set to `madvise`) and most memory is page cache, large numpy allocations can stall. If loading and grouping are unexpectedly slow (see `--profile`), try `NUMPY_MADVISE_HUGEPAGE=0`; on one shared node this made the run 2x faster. Where plenty of memory is free, huge pages help, so leave it on.
- To check a new machine: `pytest tests/test_scaling_engines.py`, and on real data `MDX2_TEST_HKL=corrected.nxs pytest -s tests/test_scaling_real_data.py`.

## Contributing

Mdx2 is being developed in collaboration with [The Diffuse Project](https://diffuse.science), and the `dev` branch in the org's fork contains the latest changes: [prism-science/mdx2/tree/dev](https://github.com/prism-science/mdx2/tree/dev).

Contributions can be made by branching from `dev`, and submitting PRs to that branch. When it is time for a new release, the `dev` branch is merged upstream to the `ando-lab/mdx2` repo.

### Developer install (conda + pip from dev branch of prism-science fork)

```bash
git clone -b dev https://github.com/prism-science/mdx2.git
cd mdx2
mamba create -f env.yaml -n mdx2-dev
mamba activate mdx2-dev
pip install -e ".[dev]"
```

The last line installs mdx2 in editable mode, with optional development tools including pytest and ruff.

CI runs the tests and ruff (pinned in the `dev` extra; notebooks are excluded). Before pushing:

```bash
ruff check . && ruff format --check .
pytest -q
```

## References

Publications describing [ando-lab/mdx2](https://github.com/ando-lab/mdx2):

- Meisburger SP & Ando N. Scaling and merging macromolecular diffuse scattering with *mdx2*. Acta Cryst. D**80**, 299-313. [DOI](https://doi.org/10.1107/S2059798324002705)
- Meisburger SP & Ando N. Chapter Two - Processing macromolecular diffuse scattering data. In *Methods in Enzymology* Volume **688**, 43-86. [DOI](https://doi.org/10.1016/bs.mie.2023.06.010), [BioRxiv](https://www.biorxiv.org/content/10.1101/2023.06.04.543637v1)

*Mdx2* is based on algorithms and general philosophy of [ando-lab/mdx-lib](https://github.com/ando-lab/mdx-lib), described here:

- Meisburger SP, Case DA & Ando N. Diffuse X-ray scattering from correlated motions in a protein crystal. *Nature Communications* **11**, 1271 (2020). [DOI](https://doi.org/10.1038/s41467-020-14933-6)
- Meisburger SP, Case DA, & Ando N. Robust total X-ray scattering workflow to study correlated motion of proteins in crystals. *Nature Communications* **14**, 1228 (2023). [DOI](https://doi.org/10.1038/s41467-023-36734-3)