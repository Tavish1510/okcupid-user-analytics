# Data

This project analyzes the [OkCupid Profiles dataset](https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles) from Kaggle (~60K dating profiles, 31 attributes including demographics, lifestyle, and 10 essay fields).

## Download Instructions

### Option 1: Kaggle CLI (recommended)

```bash
pip install kaggle

# Set up Kaggle API credentials (one-time):
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token" — downloads kaggle.json
# 3. Move kaggle.json to:
#    - Windows:  C:\Users\<you>\.kaggle\kaggle.json
#    - Mac/Linux: ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d andrewmvd/okcupid-profiles -p data/raw --unzip
```

### Option 2: Manual download

1. Go to https://www.kaggle.com/datasets/andrewmvd/okcupid-profiles
2. Click **Download** (requires free Kaggle account)
3. Unzip the file
4. Place `okcupid_profiles.csv` in `data/raw/`

## Expected file

```
data/
└── raw/
    └── okcupid_profiles.csv   (~30 MB, ~60K rows, 31 columns)
```

## Schema

| Column | Type | Description |
|---|---|---|
| age | int | User age |
| status | str | single, available, seeing someone, married |
| sex | str | m, f |
| orientation | str | straight, gay, bisexual |
| body_type | str | thin, fit, athletic, average, etc. |
| diet | str | dietary preferences |
| drinks | str | drinking frequency |
| drugs | str | drug usage |
| education | str | education level |
| ethnicity | str | comma-separated ethnicities |
| height | float | inches |
| income | int | annual income (-1 if not disclosed) |
| job | str | profession category |
| last_online | datetime | last activity timestamp |
| location | str | city, state |
| offspring | str | kids preferences |
| pets | str | dog/cat preferences |
| religion | str | religious affiliation |
| sign | str | zodiac sign |
| smokes | str | smoking frequency |
| speaks | str | comma-separated languages |
| essay0 - essay9 | str | 10 essay/self-description fields |
