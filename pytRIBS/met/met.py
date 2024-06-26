import sys
import os
import json
import urllib3
import certifi
import requests
from time import sleep
import re


def download_nldas_data(begTime, endTime, lat_range, lon_range, token, download_dir):
    """
    Downloads NLDAS data from NASA's GES DISC service within a specified bounding box and time range.

    Parameters:
    begTime (str): Beginning time in ISO 8601 format (e.g., '2024-06-20T00:00:00.000Z')
    endTime (str): Ending time in ISO 8601 format (e.g., '2024-06-20T03:59:59.999Z')
    lat_range (tuple or list): Latitude range as (minlat, maxlat)
    lon_range (tuple or list): Longitude range as (minlon, maxlon)
    token (str): Earthdata token for authentication
    download_dir (str): Path to the directory where the files will be saved

    Returns:
    None
    """

    minlat, maxlat = lat_range
    minlon, maxlon = lon_range

    # Initialize the urllib3 PoolManager
    http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

    # Set the URL for the GES DISC subset service endpoint
    svcurl = 'https://disc.gsfc.nasa.gov/service/subset/jsonwsp'

    # Define a method to POST formatted JSON WSP requests to the GES DISC endpoint URL and return the response
    def get_http_data(request):
        hdrs = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        data = json.dumps(request)
        r = http.request('POST', svcurl, body=data, headers=hdrs)
        response = json.loads(r.data)

        # Check for errors
        if response['type'] == 'jsonwsp/fault':
            print('API Error: faulty request')
        return response

    # Define the parameters for the data subset
    product = 'NLDAS_FORA0125_H_002'
    varNames = ['/HDFEOS/GRIDS/NLDAS_FORA0125_H_002/Data Fields/APCP']

    # Construct JSON WSP request for API method: subset
    subset_request = {
        'methodname': 'subset',
        'type': 'jsonwsp/request',
        'version': '1.0',
        'args': {
            'role': 'subset',
            'start': begTime,
            'end': endTime,
            'box': [minlon, minlat, maxlon, maxlat],
            'crop': True,
            'data': [{'datasetId': product}]
        }
    }

    # Submit the subset request to the GES DISC Server
    response = get_http_data(subset_request)

    # Report the JobID and initial status
    myJobId = response['result']['jobId']
    print('Job ID: ' + myJobId)
    print('Job status: ' + response['result']['Status'])

    # Construct JSON WSP request for API method: GetStatus
    status_request = {
        'methodname': 'GetStatus',
        'version': '1.0',
        'type': 'jsonwsp/request',
        'args': {'jobId': myJobId}
    }

    # Check on the job status after a brief
    sleep_time = 5
    while response['result']['Status'] in ['Accepted', 'Running']:
        sleep(sleep_time)
        response = get_http_data(status_request)
        status = response['result']['Status']
        percent = response['result']['PercentCompleted']
        print('Job status: %s (%d%c complete)' % (status, percent, '%'))

        sleep_time = min(sleep_time * 2, 60)

    if response['result']['Status'] == 'Succeeded':
        print('Job Finished: %s' % response['result']['message'])
    else:
        print('Job Failed: %s' % response['fault']['code'])
        sys.exit(1)

    # Retrieve a plain-text list of results in a single shot using the saved JobID
    result = requests.get('https://disc.gsfc.nasa.gov/api/jobs/results/' + myJobId)
    try:
        result.raise_for_status()
        urls = result.text.split('\n')

        # Create the directory to save the downloaded files
        os.makedirs(download_dir, exist_ok=True)

        # Create a session to handle authentication
        session = requests.Session()
        headers = {'Authorization': f'Bearer {token}'}

        for url in urls:
            if url.strip():  # Make sure URL is not empty
                print('\n%s' % url)
                try:
                    # Change the format in the URL to NetCDF (bmV0Q0RGLw)
                    url = url.replace('Z3JiLw', 'bmV0Q0RGLw')
                    file_response = session.get(url, headers=headers)
                    file_response.raise_for_status()

                    # Extract the date from the URL using regex
                    date_match = re.search(r'\.A(\d{8})\.', url)
                    if date_match:
                        date_str = date_match.group(1)
                        file_name = f'NLDAS_{date_str}.nc'
                    else:
                        # Fallback to a default name if the date is not found
                        file_name = 'NLDAS_data.nc'

                    file_path = os.path.join(download_dir, file_name)

                    # Save the content to a file
                    with open(file_path, 'wb') as file:
                        for chunk in file_response.iter_content(chunk_size=1000):
                            file.write(chunk)
                    print(f"Downloaded {file_name}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve results: {e}")

# Usage Example
# download_nldas_data('2024-06-20T00:00:00.000Z', '2024-06-20T03:59:59.999Z', (34.9, 37.1), (-113.8, -111.1), 'your_earthdata_token', 'nldas_downloads')
