# CHB-MIT Dataset Download Guide

## Dataset Overview

The **CHB-MIT Scalp EEG Database** is a publicly available dataset from PhysioNet containing EEG recordings from pediatric subjects with intractable seizures. The database consists of EEG recordings from 22 subjects (23 cases, since one subject was recorded twice).

- **Source**: [PhysioNet CHB-MIT](https://physionet.org/content/chbmit/1.0.0/)
- **Format**: European Data Format (EDF)
- **Sampling Rate**: 256 Hz
- **Channels**: 18-23 EEG channels (varies by subject)
- **Total Size**: ~42 GB (uncompressed)

## Download Methods

### Method 1: Manual Browser Download (Easiest)

1. Visit: https://physionet.org/content/chbmit/1.0.0/
2. Click "Files" tab
3. Download each subject folder (chb01, chb02, etc.) as ZIP
4. Extract to `data/chbmit_raw/`

**Expected structure:**
```
data/
└── chbmit_raw/
    ├── chb01/
    │   ├── chb01-summary.txt
    │   ├── chb01_01.edf
    │   ├── chb01_02.edf
    │   └── ...
    ├── chb02/
    │   ├── chb02-summary.txt
    │   └── ...
    └── ...
```

### Method 2: wget Recursive Download (Linux/Mac/WSL)

```bash
# Create data directory
mkdir -p data/chbmit_raw
cd data/chbmit_raw

# Download recursively (resume-capable)
wget -r -N -c -np -nH --cut-dirs=3 \
    https://physionet.org/files/chbmit/1.0.0/

# This may take several hours depending on connection speed
```

**Options explained:**
- `-r`: Recursive download
- `-N`: Only download if newer than local
- `-c`: Continue partial downloads
- `-np`: Don't ascend to parent directory
- `-nH`: Don't create host directory
- `--cut-dirs=3`: Remove first 3 path components

### Method 3: AWS S3 Public Mirror (Fastest)

PhysioNet mirrors data on AWS S3 with no authentication required.

```bash
# Install AWS CLI if needed
pip install awscli

# Sync entire dataset (~42 GB)
aws s3 sync --no-sign-request \
    s3://physionet-open/chbmit/1.0.0/ \
    data/chbmit_raw/

# Download specific subjects only
aws s3 sync --no-sign-request \
    s3://physionet-open/chbmit/1.0.0/chb01/ \
    data/chbmit_raw/chb01/
```

### Method 4: Google Cloud Storage Mirror

```bash
# Install gsutil if needed
pip install gsutil

# Sync dataset
gsutil -m rsync -r \
    gs://physionet-open/chbmit/1.0.0/ \
    data/chbmit_raw/
```

## Subset Download (Recommended for Testing)

For initial testing, download only a few subjects:

```bash
# Download subjects 1, 2, 3, 5, 10 (good variety)
for subj in chb01 chb02 chb03 chb05 chb10; do
    aws s3 sync --no-sign-request \
        s3://physionet-open/chbmit/1.0.0/${subj}/ \
        data/chbmit_raw/${subj}/
done
```

## Verify Download

After downloading, verify the dataset:

```bash
python scripts/verify_dataset.py --data_root data/chbmit_raw
```

Expected output should show:
- 22+ subject folders
- ~650+ EDF files total
- Seizure annotations for subjects with events

## Subject Information

| Subject | Age | Gender | Seizures | Notes |
|---------|-----|--------|----------|-------|
| chb01 | 11 | F | 7 | |
| chb02 | 11 | M | 3 | |
| chb03 | 14 | F | 7 | |
| chb04 | 22 | M | 4 | |
| chb05 | 7 | F | 5 | |
| chb06 | 1.5 | F | 10 | |
| chb07 | 14.5 | F | 3 | |
| chb08 | 3.5 | M | 5 | |
| chb09 | 10 | F | 4 | |
| chb10 | 3 | M | 7 | |
| chb11 | 12 | F | 3 | |
| chb12 | 2 | F | 40 | High seizure count |
| chb13 | 3 | F | 12 | |
| chb14 | 9 | F | 8 | |
| chb15 | 16 | M | 20 | |
| chb16 | 7 | F | 10 | |
| chb17 | 12 | F | 3 | |
| chb18 | 18 | F | 6 | |
| chb19 | 19 | F | 3 | |
| chb20 | 6 | F | 8 | |
| chb21 | 13 | F | 4 | |
| chb22 | 9 | F | 3 | |
| chb23 | 6 | F | 7 | Same as chb01 (re-recorded) |
| chb24 | - | - | 16 | Added later |

## Troubleshooting

### Download Interrupted
Use resume-capable options (`-c` for wget, `sync` for aws/gsutil).

### Checksum Verification
PhysioNet provides SHA256 checksums:
```bash
wget https://physionet.org/files/chbmit/1.0.0/SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
```

### Disk Space
Ensure at least 60 GB free:
- Raw data: ~42 GB
- Cached windows: ~15 GB
- Temporary files: ~3 GB

### Slow Download
The S3 mirror is typically fastest. If using wget, consider:
```bash
# Parallel download with aria2
aria2c -x 16 -s 16 -i urls.txt
```

## Data License

The CHB-MIT database is available under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/chbmit/view-license/1.0.0/).

**Citation required:**
```
Shoeb, A. H. (2009). Application of Machine Learning to Epileptic Seizure 
Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology.
```
