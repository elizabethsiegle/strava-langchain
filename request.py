import requests
import urllib3
from dotenv import dotenv_values
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) #I forget why I added this
config = dotenv_values(".env") #pip install chromadb, tabulate

import pandas as pd #dataframe
import re
import datetime as dt #datetime for formatting iso 8601 date
from datetime import date, timedelta #convert seconds to mins, hours, etc

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
cols = ['name', 'type', 'distance', 'moving_time']
activities = activities[cols]

# make CSV of runs
runs = activities.loc[activities['type'] == 'Run']
runs.to_csv('runs.csv', index=False) #index=False writes out weird unnamed index column in pandas df

#convert meters to miles
data_df = pd.read_csv('runs.csv')
m_conv_factor = 1609
def convert_to_miles(num):
    return ((num/m_conv_factor)*1000)/1000.0
data_df['distance'] = data_df['distance'].map(lambda x: convert_to_miles(x))
#convert moving time secs to mins, hours
#data_df['moving_time'] = data_df['moving_time'].astype(str).map(lambda x: x[7:]) #slices off 0 days from moving_time
data_df.to_csv('runs.csv')

# # make CSV of rides
# rides = activities.loc[activities['type'] == 'Ride']
# rides.to_csv('rides.csv')

# # make CSV of walks
# walks = activities.loc[activities['type'] == 'Walk']
# walks.to_csv('walks.csv')

from langchain import SerpAPIWrapper, LLMChain 
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_pandas_dataframe_agent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.chat_models import ChatOpenAI
from typing import List, Union
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
os.environ["OPENAI_API_KEY"] = config.get('OPENAI_API_KEY')

search = SerpAPIWrapper(serpapi_api_key=config.get('SERPAPI_API_KEY'))
search_tool = Tool(
    name="Search",
    func=search.run,
    description="useful for when you need to answer questions about current events",
)
# number of days for workouts
def calc_days_till_marathon(): 
    todaysDate = dt.datetime.now().strftime('%Y, %m, %d')
    date_object = dt.datetime.strptime(todaysDate, '%Y, %m, %d').date()
    d1 = date(2023, 12, 10)
    delta = d1 - date_object
    return delta
#number of workouts in Strava data
def num_rows_in_dataframe(df):
    return len(df.index)
#avg mile time in Strava data
def get_avg_mile_time(df):
    avg_miles = []
    for a,b in zip(df.distance, df.moving_time):
        avg_miles.append((b/a)/60)
    return sum(avg_miles) / len(avg_miles)

avg_distance = data_df['distance'].mean()
avg_moving_time = data_df['moving_time'].mean()
avg_miles = []
for a,b in zip(data_df.distance, data_df.moving_time):
    avg_miles.append((b/a)/60)
    avg_mile= sum(avg_miles) / len(avg_miles)
print(avg_distance)
print(avg_moving_time)
print(avg_mile)
marathon_date = "December, 10, 2023" #hard-coded marathon_date

tools = [
    Tool(
        name = "today",
        func = lambda x: calc_days_till_marathon(),
        description="use to get today's date"
    ),
    Tool(
        name = "rows in csv",
        func = lambda df: num_rows_in_dataframe(df),
        description="use to get the number of rows in csv file to calculate averages from running data"
    ),
    search_tool
]
llm = ChatOpenAI(temperature=0)
pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), data_df, verbose=True) #csv agent?
# the CSV Agent can be used to load data from CSV files and perform queries, while the Pandas Agent can be used to load data from Pandas data frames and process user queries. Agents can be chained together to build more complex applications.

agent_output_day_answer =calc_days_till_marathon() #pd_agent.run("as an integer, how many days from today until my marathon on December 10, 2023") #with tool for dates
print("days to marathon: ", agent_output_day_answer)
num_rows_in_dataframe = num_rows_in_dataframe(data_df)
pd_agent.run("Calculate Lizzie's average distance, average moving time converted to minutes, and quartiles of distance and moving time. Calculate other statistics you think would be helpful for marathon training based on the data in {data_df}.")

coach_template = """You are a personal marathon trainer. You know a lot about your student, like her previous runs from the past few months, her average mile time, her average moving time, and up until today. 
Here is some context about the time and location of the marathon she is training for:
Today: {agent_output_day_answer}
Marathon date: {marathon_date}
Her average distance each time she runs:{avg_distance}
Her average moving time of each run: {avg_moving_time}
Her average mile time: {avg_mile}
Make her a marathon training plan to follow, including a list of days between {agent_output_day_answer} and {marathon_date} with a run or workout. The first day should not be a rest day.
There should be {agent_output_day_answer} workouts that get slightly progressively longer so she can be ready for her marathon, but she can't get injured or burned out so there should be no more than one long run each week. 
A long run should not be longer than 20 miles. The plan should also include rest days, sprints, and can include cross-training like bike rides and swimming. After each day, start the next day's workout on a new line  You have access to the following tools:
{tools}
Running calendar plan with days of the month:
"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
prompt = CustomPromptTemplate(
    template=coach_template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["agent_output_day_answer", "marathon_date", "avg_mile","avg_distance", "avg_moving_time", "intermediate_steps"]
)
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Day 1:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()
# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
plan = agent_executor({"agent_output_day_answer":agent_output_day_answer, "marathon_date":marathon_date, "avg_mile" :avg_mile, "avg_distance" :avg_distance, "avg_moving_time": avg_moving_time})

message = Mail(
    from_email='langchain_sendgrid_marathon_trainer@sf.com',
    to_emails='lizzie.siegle@gmail.com',
    subject='Your AI-generated marathon training plan',
    html_content='<strong>Good luck at your marathon on %s</strong>\n\nHere is your plan:\n\n%s'%(marathon_date, plan['output']))

os.environ["SENDGRID_API_KEY"] = config.get('SENDGRID_API_KEY')
sg = SendGridAPIClient()
response = sg.send(message)
code, body, headers = response.status_code, response.body, response.headers
print(f"Response Code: {code} ")
print(f"Response Body: {body} ")
print(f"Response Headers: {headers} ")
print("Message Sent!")