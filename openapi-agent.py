from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.openapi import planner
import yaml
from dotenv import dotenv_values

with open("swagger.yml") as f:
    raw_openai_api_spec = yaml.load(f, Loader=yaml.Loader)

openai_api_spec = reduce_openapi_spec(raw_openai_api_spec)
config = dotenv_values(".env")
strava_token = config.get('STRAVA_TOKEN')
headers = {
    "Authorization": "Bearer {strava_token}".format(strava_token = strava_token),
    "Content-Type": "application/json"
}
# Get API credentials.
requests_wrapper = RequestsWrapper(headers=headers)

llm = ChatOpenAI(openai_api_key=config.get('OPENAI_API_KEY'), model_name="gpt-4", temperature=0.0)
openai_agent = planner.create_openapi_agent(openai_api_spec, requests_wrapper, llm)

query = "how far was her last activity of type run? how many minutes did it take her?"
openai_agent.run(query) #it knows last is most recent

