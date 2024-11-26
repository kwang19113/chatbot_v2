from langchain.llms.base import LLM
from pydantic import Field
from typing import Optional
from langchain import hub
from langchain_community.document_loaders import WikipediaLoader
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from General import generate_query,generate_response,get_SQL_quyery


# Define a custom LLM wrapper
class CustomLLM(LLM):
    generate_response_fn: callable = Field()

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        # Use the generate_response function to get the response
        response = self.generate_response_fn(prompt)
        return response

import requests
import json

def generate_response(prompt):
    url = 'http://localhost:11434/api/generate'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'model': 'gemma2:27b',
        'prompt': prompt,
        'stream': False,
        #'system_prompt': "Your system prompt here",  # Replace as needed
        'temperature': 0.7  # Replace as needed
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get('response', 'No response received.')
    except requests.exceptions.RequestException as e:
        return f'An error occurred: {e}'

# Initialize the custom LLM with the generate_response function
custom_llm = CustomLLM(generate_response_fn=generate_response)


from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.tools import tool
import wikipediaapi
import math
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic import SkipValidation

# Replace any usage of callable with SkipValidation
# Example:
my_field: SkipValidation = callable  # Correct usage




# Define a custom tool for performing calculations
@tool("Calculator")
def calculate(expression: str) -> str:
    """Perform mathematical calculations for the given expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": None}, math.__dict__)
        return str(result)
    except Exception as e:
        return f"Error in calculation: {e}"



class Agent:
    def __init__(self, client, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = custom_llm.invoke(self.messages)
        return completion

system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

search:
e.g. search: albert einstein
returns information about albert einstein

get_SQL_quyery:
e.g. get_SQL_quyery: find people wearing red top
returns a query from the given information

execute_query:
e.g. execute_query: SELECT * FROM par_table WHERE upper_color = 'red'
return a list of results

Example session:
Question: How old is albert einstein in 1900
Thought: I need to find albert einstein born year
Action: search: albert einstein
PAUSE 

You will be called again with this:

Observation: Albert Einstein (/ˈaɪnstaɪn/ EYEN-styne,[5] German: [ˈalbɛʁt ˈʔaɪnʃtaɪn] ⓘ; 14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held as one of the most influential scientists. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics.[1][6] His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called "the world's most famous equation".[7] He received the 1921 Nobel Prize in Physics.[8]

Thought: He was born in 1879. i need to minus 1900 with 1879
Action: calculate: 1900 - 1879
PAUSE

You will be called again with this: 

Observation: 21

If you have the answer, output it as the Answer.

Answer: He was 21 years old in 1900.


Example session 2:

Question: find people wearing red top
Thought: I need to get a query to communicate with the database
Action: get_SQL_quyery: find people wearing red top
PAUSE
You will be called again with this:

Observation: sql: SELECT * FROM par_table WHERE upper_color = 'red'

Thought: I need to execute SELECT * FROM par_table WHERE upper_color = 'red' and get the information
Action: execute_query: SELECT * FROM par_table WHERE upper_color = 'red'
PAUSE

You will be called again with this: (6, 'short_sleeve', 'trouser', 'red', 'beige', 'bare_head', 'male', '11-26-2024', 'Screenshot 2024-07-04 161135.png')
(17, 'short_sleeve', 'trouser', 'red', 'black', 'hat', 'male', '02-08-2024', 'Screenshot 2024-07-04 161225.png')
(52, 'short_sleeve', 'trouser', 'red', 'black', 'bare_head', 'male', '02-22-2024', 'Screenshot 2024-07-04 161514.png')
(54, 'short_sleeve', 'trouser', 'red', 'blue', 'bare_head', 'male', '11-17-2024', 'Screenshot 2024-07-04 161525.png')

If you have the answer, output it as the Answer.

Answer: 
Five individuals wearing red tops were observed in different images across 2024:
ID 6: Male, red short sleeve, beige trousers, bare head (11-26-2024).
ID 17: Male, red short sleeve, black trousers, hat (02-08-2024).
ID 52: Male, red short sleeve, black trousers, bare head (02-22-2024).
ID 54: Male, red short sleeve, blue trousers, bare head (11-17-2024).
ID 60: Female, red short sleeve, black trousers, bare head (03-18-2024).
Four males and one female were observed; four were bare-headed, and trousers varied in beige, black, and blue.

Now it's your turn:

""".strip()

import sqlite3
def calculate(operation: str) -> float:
    return eval(operation)

def search(query: str):
    engine = DuckDuckGoSearchRun()
    result = engine.invoke(query)
    return result

def execute_query(query: str):
    conn = sqlite3.connect("../database/chat.db")
    cursor = conn.cursor()
    cursor.execute(query)
    result_string = ""
    # Fetch and print the matching rows
    rows = cursor.fetchall()
    for row in rows:
        print(row)
        result_string = '\n'.join([str(row) for row in rows])
    conn.close()
    if result_string == "":
        return "no data found"
    return result_string
neil_tyson = Agent(client=custom_llm, system=system_prompt)
import re


def loop(max_iterations=10, query: str = ""):

    agent = Agent(client=custom_llm, system=system_prompt)

    tools = ["calculate", "search", "get_SQL_quyery", "execute_query"]

    next_prompt = query

    i = 0
  
    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        print(result)

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            chosen_tool = action[0][0]
            arg = action[0][1]

            if chosen_tool in tools:
                print(f"{chosen_tool}('{arg}')")
                result_tool = eval(f'{chosen_tool}("""{arg}""")')
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            print(next_prompt)
            continue

        if "Answer" in result:
            break


loop(query="What is the capital of the largest country by population in Europe?")