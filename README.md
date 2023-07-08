### Query Strava data with LangChain (and Twilio) and then use that data to generate a personal marathon training plan with SendGrid

#### 
- `data_files` has Strava data in different files.
    - `activities.csv` are my workouts from Strava. They include all my workouts from 2021 to present and the workouts have the following columns: name,type,distance,moving_time,total_elevation_gain,start_date
    - `rides.csv` has my workouts filtered for type <em>ride</em>       - `runs.csv` has my workouts filtered for type <em>run</em>        - `walks.csv` has my workouts filtered for type <em>run</em>
- `plans` has marathon training plans.
    - `plan.pdf` is an AI-generated workout plan for me based on my Strava data.
    - `20-Weeks-Marathon-Training-Plan-Miles.pdf` is a marathon plan I got online
    - `Marathon.pdf`is a 16-week Marathon Training Schedule for Novice Runners
- `strava-token.py` generates a Strava API token
- `swagger.yml` has OpenAPI specification for the Strava API
- `csv-agent-pd-agent.py` uses a Pandas dataframe agent to query Strava data from `activities.csv` with a Pandas Dataframe.
- `openapi-agent.py` reads `swagger.yml` and makes queries to Strava with OpenAPISpec Agent
- `openapi-chain.py` reads `swagger.yml` and makes queries to Strava with OpenAPISpec Chain
- `request.py` generates a personal marathon training plan using SendGrid, Pandas Dataframe agent, tools like calculating the current date and Search, a CustomPromptTemplate, CustomOutputParser, LLMSingleActionAgent, and AgentExecutor.  