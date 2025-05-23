# wildfire_prediction

### Folder structure
```
.
├── app.py
├── data
│   ├── combined_data.csv
│   └── processed
│       ├── korea_combined.csv
│       ├── nepal_combined.csv
│       ├── processed_modis_2020_Nepal.csv
│       ├── processed_modis_2020_Republic_of_Korea.csv
│       ├── processed_modis_2021_Nepal.csv
│       ├── processed_modis_2021_Republic_of_Korea.csv
│       ├── processed_modis_2022_Nepal.csv
│       ├── processed_modis_2022_Republic_of_Korea.csv
│       ├── processed_modis_2023_Nepal.csv
│       └── processed_modis_2023_Republic_of_Korea.csv
├── datasets
│   ├── DL_FIRE_J1V-C2_601036
│   │   ├── fire_archive_J1V-C2_601036.csv
│   │   └── Readme.txt
│   ├── DL_FIRE_J1V-C2_601038
│   │   ├── fire_archive_J1V-C2_601038.csv
│   │   └── Readme.txt
│   ├── DL_FIRE_SV-C2_601037
│   │   ├── fire_archive_SV-C2_601037.csv
│   │   └── Readme.txt
│   ├── DL_FIRE_SV-C2_601039
│   │   ├── fire_archive_SV-C2_601039.csv
│   │   └── Readme.txt
│   ├── modis_2021_all_countries.zip
│   ├── modis_2022_all_countries.zip
│   └── modis_2023_all_countries.zip
├── LICENSE
├── models
│   ├── wildfire_predictor_model.pkl
│   └── wildfire_predictor_scaler.pkl
├── notebooks
│   ├── data_exploration.ipynb
│   └── model_training.ipynb
├── README.md
└── src
    ├── data_preprocessing.py
    ├── feature_engineering.py
    ├── model.py
    └── utils.py
```