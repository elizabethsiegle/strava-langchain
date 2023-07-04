import requests
import urllib3
from dotenv import dotenv_values
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) #I forget why I added this
config = dotenv_values(".env") 

import pandas as pd #dataframe

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

#Create new dataframe with specific columns #max_time
cols = ['name', 'type', 'distance', 'moving_time', 'total_elevation_gain', 'start_date']
activities = activities[cols]
activities = activities[activities["start_date"].str.contains("2021") == False] #remove items from 2021, only include workouts from 2022 and 2023
activities.to_csv('activities.csv', index=False)

# loop through activities data frame to get number of activities of each type
num_runs = len(activities.loc[activities['type'] == 'Run'])
num_walks = len(activities.loc[(activities['type'] == 'Walk') & (activities['total_elevation_gain'] > 90)])
num_rides = len(activities.loc[activities['type'] == 'Ride'])
num_elliptical = len(activities.loc[activities['type'] == 'Elliptical'])
num_weight_training = len(activities.loc[activities['type'] == 'WeightTraining'])
num_swims = 0
num_tennis = 0
for i in activities['name'].values:
    if 'swim' in i.lower():
        num_swims +=1
    if 'tennis' in i.lower():
        num_tennis +=1
cross_training_options = activities['type'].unique()
print('num_runs ', num_runs)
print('num_walks ', num_walks)
print('num_rides ', num_rides)
print('num_elliptical ', num_elliptical)
print('num_weight_training ', num_weight_training)
print('num_swims ', num_swims)
print('num_tennis ', num_tennis)

from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import Tool, create_pandas_dataframe_agent, create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.utilities.twilio import TwilioAPIWrapper
import os
os.environ["OPENAI_API_KEY"] = config.get('OPENAI_API_KEY')

search = SerpAPIWrapper(serpapi_api_key=config.get('SERPAPI_API_KEY'))
search_tool = Tool(
    name="Search",
    func=search.run,
    description="useful for when you need to answer questions about current events",
)
data_df = pd.read_csv('runs.csv')
llm = ChatOpenAI(temperature=0)
pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data_df, verbose=True) #csv agent?
# csv agent can be used to load data from CSV files and perform queries, while the Pandas Agent can be used to load data from Pandas data frames and process user queries. Agents can be chained together to build more complex applications.
pd_agent.run("how many runs, bike rides, swims, and walks with total elevation gain over 100 has Lizzie done")
pd_agent.run("how many runs has Lizzie done")
pd_agent.run("how many bike rides has Lizzie done? How many swims?")
pd_agent.run("What was Lizzie's fastest half marathon? Fastest 10k?")
pd_agent.run("How many walks with total elevation gain over 100 has she taken?")
pd_agent.run("How fast did Lizzie run the Monterey half marathon last year?")
csv_agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
   "activities.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
csv_agent.run("how many runs has Lizzie done")
csv_agent.run("how many bike rides has Lizzie done? How many swims?")
csv_agent.run("What was Lizzie's fastest half marathon?")
csv_agent.run("How many walks with total elevation gain over 100 has she taken?")
csv_multi_agent = create_csv_agent(
    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
    ["activities.csv", "runs.csv"],
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
csv_multi_agent.run("how many rows in the moving_time column are different between the two dfs?")
twilio = TwilioAPIWrapper(
    account_sid=os.environ['TWILIO_ACCOUNT_SID'],
    auth_token=os.environ['TWILIO_AUTH_TOKEN'],
    from_number=os.environ['MY_TWILIO_NUMBER']
)
twilio.run("hello world", os.environ['MY_PHONE_NUMBER'])