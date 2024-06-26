import sys
import os
import shutil
import json
import urllib3
import certifi
import requests
from time import sleep
import re
import xarray as xr


class _Met:
    @staticmethod
    def get_nldas_data(begin, end, lat_range, lon_range, token, download_dir, verbose = False):
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

        Note: There is currently an issue where the following exception occurs: Sorry,
        NLDAS_FORA0125_H.A20200620.0000.002.grb.SUB.grb is not available for downloading. Currently, why this occurs has
        not been identified. Waiting a few minutes and running the function again the exception will disapper.
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
            http_response = json.loads(r.data)

            # Check for errors
            if http_response['type'] == 'jsonwsp/fault':
                print('API Error: faulty request')
            return http_response

        # Define the parameters for the data subset
        product = 'NLDAS_FORA0125_H_002'

        # Construct JSON WSP request for API method: subset
        subset_request = {
            'methodname': 'subset',
            'type': 'jsonwsp/request',
            'version': '1.0',
            'args': {
                'role': 'subset',
                'start': begin,
                'end': end,
                'box': [minlon, minlat, maxlon, maxlat],
                'crop': True,
                'data': [{'datasetId': product}]
            }
        }

        # Submit the subset request to the GES DISC Server
        response = get_http_data(subset_request)

        # Report the JobID and initial status
        my_job_id = response['result']['jobId']

        if verbose:
            print('Obtaining links')
            print('Job ID: ' + my_job_id)
            print('Job status: ' + response['result']['Status'])

        # Construct JSON WSP request for API method: GetStatus
        status_request = {
            'methodname': 'GetStatus',
            'version': '1.0',
            'type': 'jsonwsp/request',
            'args': {'jobId': my_job_id}
        }

        # Check on the job status after a brief
        sleep_time = 5
        while response['result']['Status'] in ['Accepted', 'Running']:
            sleep(sleep_time)
            response = get_http_data(status_request)
            status = response['result']['Status']
            percent = response['result']['PercentCompleted']

            if verbose:
                print('Job status: %s (%d%c complete)' % (status, percent, '%'))

            sleep_time = min(sleep_time * 2, 60)

        if response['result']['Status'] == 'Succeeded':
            if verbose:
                print('Job Finished: %s' % response['result']['message'])
        else:
            print('Job Failed: %s' % response['fault']['code'])
            sys.exit(1)

        # Retrieve a plain-text list of results in a single shot using the saved JobID
        result = requests.get('https://disc.gsfc.nasa.gov/api/jobs/results/' + my_job_id)
        result.raise_for_status()
        urls = result.text.split('\n')

        # Create the directory to save the downloaded files
        os.makedirs(download_dir, exist_ok=True)

        # Create a session to handle authentication
        session = requests.Session()
        headers = {'Authorization': f'Bearer {token}'}

        if verbose:
            print("Downloading Files... this may take a while")

        for url in urls:
            if url.strip():  # Make sure URL is not empty
                print('\n%s' % url)
                try:

                    # Change the format in the URL to NetCDF (bmM0Lw)
                    url = url.replace('Z3JiLw', 'bmM0Lw')
                    file_response = session.get(url, headers=headers)
                    file_response.raise_for_status()

                    # Extract the date from the URL using regex
                    date_match = re.search(r'\.A(\d{8}\.\d{4})\.', url)
                    if date_match:
                        date_str = date_match.group(1)
                        year = date_str[:4]
                        file_name = f'NLDAS_{date_str}.nc'
                    else:
                        # Fallback to a default name if the date is not found
                        print("There was an issue with file name")
                        exit(1)

                    # Create a directory for the year if it doesn't exist
                    year_dir = os.path.join(download_dir, year)
                    os.makedirs(year_dir, exist_ok=True)

                    file_path = os.path.join(year_dir, file_name)

                    # Save the content to a file
                    with open(file_path, 'wb') as file:
                        for chunk in file_response.iter_content(chunk_size=1000):
                            file.write(chunk)
                    if verbose:
                        print(f"Downloaded {file_name}")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to download {url}: {e}")
    @staticmethod
    def merge_nldas_files_by_year(download_dir, remove_hourly=True):
        """
        Merge NetCDF files for each year found in subdirectories of the specified download directory.

        Parameters:
        download_dir (str): Path to the directory containing subdirectories labeled with years.
        remove_hourly (bool): Whether to remove the hourly directories after merging (default: True).

        Returns:
        None
        """
        for year in os.listdir(download_dir):
            year_dir = os.path.join(download_dir, year)
            if os.path.isdir(year_dir):
                try:
                    # Collect all files for the year
                    files = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.nc')]
                    # Load and merge the files
                    ds = xr.open_mfdataset(files, combine='by_coords')
                    merged_file = os.path.join(download_dir, f'NLDAS_{year}_merged.nc')
                    ds.to_netcdf(merged_file)

                    # Check number of time steps in merged dataset
                    num_time_steps = len(ds.time)

                    # Compare with number of files
                    num_files = len(files)
                    if num_time_steps == num_files:
                        if remove_hourly:
                            shutil.rmtree(year_dir, ignore_errors=True)
                            print(f"Merged files for year {year} into {merged_file}. Removed {year_dir}.")
                        else:
                            print(f"Merged files for year {year} into {merged_file}.")
                    else:
                        print(
                            f"Warning: Number of time steps ({num_time_steps}) does not match number of files ({num_files}) for year {year}.")
                        # Optionally handle mismatch here, e.g., log warning or skip removal

                except Exception as e:
                    print(f"Failed to merge files for year {year}: {e}")


