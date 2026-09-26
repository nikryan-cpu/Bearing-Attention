# Bearing Attention

A digital stethoscope for bearings: detecting bearing wear early from vibration data with a small forecasting transformer.

Work in progress.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The extra index only makes pip pick the CPU build of PyTorch on Linux; nothing here needs a GPU.

## Data

The raw data is not part of this repository (the license allows use with citation but not redistribution). This downloads both archives (about 2.2 GB) from the PHM Society mirror of the NASA Prognostics Data Repository and unpacks them into `data/raw/`:

```
python scripts/download_data.py
```

The download resumes if the connection drops. If you already have the zips, pass them with `--femto-archive` / `--ims-archive`. The IMS data is packed as .7z and .rar, so it needs bsdtar (built into Windows 10+ and macOS, `apt install libarchive-tools` on Debian/Ubuntu) or 7-Zip.

- FEMTO: P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Morello, N. Zerhouni, C. Varnier. PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test. IEEE International Conference on Prognostics and Health Management, Denver, CO, USA, 2012. Data: "FEMTO Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- IMS: J. Lee, H. Qiu, G. Yu, J. Lin, and Rexnord Technical Services (2007). IMS, University of Cincinnati. "Bearing Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
