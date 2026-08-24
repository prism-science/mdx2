# Data processing tutorial: insulin

This tutorial demonstrates the essential steps of reciprocal space mapping with `mdx2` on a small dataset from cubic insulin.

The tutorial is based on the [data processing workshop](https://github.com/ando-lab/erice-2022-data-reduction) at the [2022 Erice School on Diffuse Scattering](https://crystalerice.org/2022/programme_ds.php) and it accompanies a chapter on data processing in the forthcoming Methods in Enzymology volume "Crystallography of Protein Dynamics”.

To begin, download the jupyter notebooks individually or clone the _mdx2_ repository. Then, follow the instructions below to download the dataset and create a stand-alone Python environment with all of the software used in the tutorial.

The code is designed to run on a personal computer with ~20 Gb of free disk storage and at least 4 Gb of RAM. A unix-like operating system is assumed (Linux, OSX, or Windows Subsystem for Linux).

### Downloading the tutorial dataset

The dataset from insulin is available on Zenodo (<https://dx.doi.org/10.5281/zenodo.6536805>). First, download `insulin_2_1.tar` and extract the tar archive. The file will expand to a directory called `images` with subfolders `insulin_2_1`, `insulin_2_bkg`, and `metrology`. Place `images` in the same directory as the Jupyter notebooks.

### Installing the software

We'll create a single `conda` environment containing a stand-alone installation of `dials`, `nexpy`, `jupyter lab`, and `mdx2`. If you already have `conda` or `mamba` installed, you can use that. If not, we recommend using `micromamba` to create the environment.

Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

The following will create the `conda` environment with the minimal dependencies for `mdx2`:

```bash
micromamba create -f https://raw.githubusercontent.com/prism-science/mdx2/main/env.yaml
```

Answer yes (`Y`) when prompted.

Next, activate the environment and install `dials`, `nexpy`, `jupyter lab`, and `mdx2`:

```bash
micromamba activate mdx2
micromamba install -c conda-forge dials nexpy jupyterlab
pip install git+https://github.com/prism-science/mdx2
```

Answer yes (`Y`) when prompted.

The _mdx2_ tools should now be available at the command line. Check the version as follows:

```bash
mdx2.version
```

With the _mdx2_ conda environment active, _nexpy_ can be launched from the command line by typing `nexpy`. Similarly, Jupyter Lab can be launched by typing `jupyter lab`. The dials command-line tools are also available. For instance `dials.version` prints the version information and install location.
