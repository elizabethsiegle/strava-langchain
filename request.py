# TODO: check how to input data/stats/facts into LLM without passing vars to template as input vars
import requests
import urllib3
from dotenv import dotenv_values
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) #I forget why I added this
config = dotenv_values(".env") #pip install chromadb, tabulate

import pandas as pd #dataframe
import numpy as np
import re
import datetime as dt #datetime for formatting iso 8601 date
from datetime import date #convert seconds to mins, hours, etc
from dateutil import rrule

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
activities.to_csv('data_files/activities.csv', index=False)

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

from langchain import SerpAPIWrapper, LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_pandas_dataframe_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI, GPT4All
from langchain.prompts import BaseChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage

from typing import List, Union
import os
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
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
max_distance_ran = data_df['distance'].max()
avg_miles = []
for a,b in zip(data_df.distance, data_df.moving_time):
    avg_miles.append((b/a)/60)
    avg_mile= sum(avg_miles) / len(avg_miles)
print('avg_distance ', avg_distance)
print('avg_moving_time ', avg_moving_time)
print('avg_mile', avg_mile)
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
# csv agent can be used to load data from CSV files and perform queries, while the Pandas Agent can be used to load data from Pandas data frames and process user queries. Agents can be chained together to build more complex applications.

agent_output_day_answer =calc_days_till_marathon() #pd_agent.run("as an integer, how many days from today until my marathon on December 10, 2023") #with tool for dates
print("days to marathon: ", agent_output_day_answer)
num_rows_in_dataframe = num_rows_in_dataframe(data_df)
pd_output = pd_agent.run("Calculate Lizzie's average distance, average moving time converted to minutes, maximum moving time, maximum disrance, and the number of times she's run, ridden her bike, played tennis, weight trained, and swam. Calculate other statistics you think would be helpful for marathon training in order to create a personalized marathon training plan for a well-rounded athlete who has never trained for a half-marathon.")
coach_template = """You are a personal marathon trainer. You know a lot about your student, like her previous runs from the past few months, her average mile time, her average moving time, and up until today. 
Here is some context about the marathon you will help her train for:
Marathon date: {marathon_date}
Her goal is to finish under 4 hours and 48 minutes at a mile pace under 11 minutes.
The tools you can use:
{tools}
Consider her average distance, average moving time, maximum distance, maximum moving distance, number of times she's run, biked, walked, swam, or done weight training and the frequency: {pd_output}
Use that data to shape her cross-training in her marathon training plan.
Total miles run weekly should eventually be around 45 miles. The number of miles run weekly should gradually increase over time.
For each week, include the total number of miles to be run for that week.
The longest run should be around 20 miles and take place 2 weeks before {marathon_date}. 
Workouts should be easy right before {marathon_date} 
For each running distance you recommend, also suggest easy, medium, or hard pace based on her average running times and the number of recommended miles. 
There should be {agent_output_day_answer} workouts starting with today that get slightly progressively longer so she can be ready for her marathon, but she can't get injured or burned out so there should be no more than one long run (a long run is any run over 10 miles) each week. 
The plan should also include sprints and cross-training like {cross_training_options}. She likes tennis and biking! After each day, start the next day's workout on a new line and include the date. She should have at least 2 cross-training workouts a week.
Running calendar plan with the start date, end date (which is the marathon on {marathon_date}), some speed workouts or cross-training with times, and goal times:
Marathon Day (December 10, 2023):
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
    input_variables=["agent_output_day_answer", "marathon_date", "cross_training_options", "pd_output", "intermediate_steps"]
)
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Marathon Day" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Marathon Day:")[-1].strip()},
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
    stop=["\Final Answer:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
plan = agent_executor({"agent_output_day_answer":agent_output_day_answer, "marathon_date":marathon_date, "cross_training_options": cross_training_options, "pd_output": pd_output})

message = Mail(
    from_email='langchain_sendgrid_marathon_trainer@sf.com',
    to_emails='lizzie.siegle@gmail.com',
    subject='Your AI-generated marathon training plan',
    html_content='<strong>Good luck at your marathon on %s</strong>!\n\nYour plan is attached.'%(marathon_date))

styles = getSampleStyleSheet()
styleN = styles['Normal']
styleH = styles['Heading1']
story = []

pdf_name = 'plan.pdf'
doc = SimpleDocTemplate(
    pdf_name,
    pagesize=letter,
    bottomMargin=.4 * inch,
    topMargin=.6 * inch,
    rightMargin=.8 * inch,
    leftMargin=.8 * inch)


P = Paragraph(plan['output'], styleN)
story.append(P)

doc.build(
    story,
)
with open(pdf_name, 'rb') as f:
    data = f.read()
    f.close()
encoded_file = base64.b64encode(data).decode()

attachedFile = Attachment(
    FileContent(encoded_file),
    FileName('attachment.pdf'),
    FileType('application/pdf'),
    Disposition('attachment')
)
message.attachment = attachedFile
os.environ["SENDGRID_API_KEY"] = config.get('SENDGRID_API_KEY')
sg = SendGridAPIClient()
response = sg.send(message)
code, body, headers = response.status_code, response.body, response.headers
print(f"Response Code: {code} ")
print(f"Response Body: {body} ")
print(f"Response Headers: {headers} ")
print("Message Sent!")