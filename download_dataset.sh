mkdir dataset
cd dataset

# Download uci_har_dataset
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip --output ./uci_har_dataset.zip
unzip uci_har_dataset.zip
mv UCI\ HAR\ Dataset uci_har_dataset
rm -rf __MACOSX