"""

    Simple Script to test the API once deployed

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located at the root of this repo for guidance on how to use this
    script correctly.
    ----------------------------------------------------------------------

    Description: This file contains code used to formulate a POST request
    which can be used to develop/debug the Model API once it has been
    deployed.

"""

# Import dependencies
import requests
import pandas as pd

# Load data from file to send as an API POST request
test_data = pd.read_csv('./data/df_test.csv')

# Convert the test data to JSON string
feature_vector_json = test_data.iloc[1].to_json()

# Specify the URL at which the API will be hosted
url = 'http://34.242.3.19:5000/api_v0.1'

# Perform the POST request
try:
    print(f"Sending POST request to web server API at: {url}")
    print(f"Querying API with the following data: \n {test_data.iloc[1].tolist()}")
    api_response = requests.post(url, json=feature_vector_json)

    # Check if the API response is successful (status code 200)
    if api_response.status_code == 200:
        # Try to parse the response as JSON
        try:
            prediction_result = api_response.json()
            print("Received POST response:")
            print("*" * 50)
            print(f"API prediction result: {prediction_result}")
            print(f"The response took: {api_response.elapsed.total_seconds()} seconds")
            print("*" * 50)
        except ValueError:
            print("Error: Invalid JSON response from the API")
    else:
        print(f"Error: API request failed with status code {api_response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"Error: Failed to connect to the API - {e}")

