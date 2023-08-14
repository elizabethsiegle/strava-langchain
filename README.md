### Query Strava data with LangChain (and Twilio) and then use that data to generate a personal marathon training plan with SendGrid

#### 
- `strava-token.py` generates a Strava API token
- `request.py` generates a personal marathon training plan using SendGrid, Pandas Dataframe agent, tools like calculating the current date and Search, a CustomPromptTemplate, CustomOutputParser, LLMSingleActionAgent, and AgentExecutor.  