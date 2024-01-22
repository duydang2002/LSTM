import pickle

# Specify the filename (DATA_FILE) where you saved the data
DATA_FILE = 'traindata.pkl'  # Replace with the actual filename

# Read the data from the file
with open(DATA_FILE, 'rb') as file:
    loaded_data = pickle.load(file)

# Unpack the zip object
labels, domains = zip(*loaded_data)  

# Print labels and domains in parallel
for label, domains  in zip(labels, domains ):
    print(f"Label: {label} Domains : {domains} " )
