import requests
import numpy as np

# Define API URL
url = "http://127.0.0.1:5000/predict"

# Generate a 28x28 black image (all zeros)
test_data = np.zeros((28, 28), dtype=int).tolist()

# Send the request
response = requests.post(url, json={"data": test_data})

# Print response
print(response.json())

