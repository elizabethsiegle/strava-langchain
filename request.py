import requests
import urllib3
from dotenv import dotenv_values
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) #I forget why I added this
config = dotenv_values(".env")

import pandas as pd #dataframe
import datetime as dt #datetime for formatting iso 8601 date
from datetime import timedelta #convert seconds to mins, hours, etc

activities_url = "https://www.strava.com/api/v3/athlete/activities"

header = {'Authorization': 'Bearer ' + config.get('STRAVA_TOKEN')}
params = {'per_page': 200, 'page': 1} #max 200 per page, can only do 1 page at a time
my_dataset = requests.get(activities_url, headers=header, params=params).json() #activities 1st page
page = 0
for x in range(1,5): #loop through 4 pages of strava activities
    page +=1 
    params = {'per_page': 200, 'page': page}
    my_dataset += requests.get(activities_url, headers=header, params=params).json() #add to dataset, need strava token in .env to be updated else get dict error
    
activities = pd.json_normalize(my_dataset)
# print(activities.columns) # list all columns in the table
# print(activities.shape) #dimensions of the table.

#Create new dataframe with only columns I care about #max_time
cols = ['name', 'type', 'distance', 'moving_time',   
         'total_elevation_gain', 'start_date_local'
       ]
activities = activities[cols]

# make CSV of runs
runs = activities.loc[activities['type'] == 'Run']
runs.to_csv('runs.csv', index=False) #index=False writes out weird unnamed index column in pandas df

#convert meters to miles
data_df = pd.read_csv('runs.csv')
m_conv_factor = 1609
data_df['distance'] = data_df['distance'].apply(lambda m: int((m / m_conv_factor)*1000)/1000.0)

#convert moving time secs to mins, hours
data_df['moving_time'] = data_df['moving_time'].apply(lambda sec: timedelta(seconds=sec))
data_df['moving_time'] = data_df['moving_time'].astype(str).map(lambda x: x[7:]) #slices off 0 days from moving_time
data_df.to_csv('runs.csv')

#convert start_date_local
def convert_iso8601(iso8601date):
    date = dt.datetime.fromisoformat(iso8601date)
    return date.strftime('%b %d, %Y')

data_df['start_date_local'] = data_df['start_date_local'].apply(lambda iso861: convert_iso8601(iso861))
data_df.to_csv('runs.csv')

# make CSV of rides
rides = activities.loc[activities['type'] == 'Ride']
rides.to_csv('rides.csv')

# make CSV of walks
walks = activities.loc[activities['type'] == 'Walk']
walks.to_csv('walks.csv')