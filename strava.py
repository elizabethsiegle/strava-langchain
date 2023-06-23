from langchain.llms import OpenAI
from langchain.requests import Requests
from langchain.chains import OpenAPIEndpointChain
from langchain.tools import OpenAPISpec, APIOperation
from dotenv import dotenv_values

config = dotenv_values(".env")

spec = OpenAPISpec.from_file("swagger.yml")
# spec = OpenAPISpec.from_url(
#     "https://developers.strava.com/swagger/swagger.json"
# )
operation = APIOperation.from_openapi_spec(spec, '/athlete/activities', "get")

strava_token = config.get('STRAVA_TOKEN')
headers = {
    "Authorization": "Bearer {strava_token}".format(strava_token = strava_token),
    "Content-Type": "application/json"
}

llm = OpenAI(openai_api_key=config.get('OPENAI_API_KEY'), temperature=0)
api_chain = OpenAPIEndpointChain.from_api_operation(
    operation, 
    llm, 
    requests=Requests(headers=headers), 
    verbose=True,
    return_intermediate_steps=True
)
print(api_chain("How many miles did the user run in her last activity?"))

