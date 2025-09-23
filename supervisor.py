import os
import requests
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

web_search = TavilySearch(max_results=3)

research_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

def get_places(query: str, latitude: float, longitude: float) -> dict:
    """Function that leverages the Google Places API to find locations near the latidute and longitude given.

    Args:
        query: The type of location you are searching for (in example "Recycling center")
        latitude: The current latitude of the user
        longitude: The current longitude of the user

    Returns:
        Dictionary with location details, results and metadata
    """
    #the google api url
    url = 'https://places.googleapis.com/v1/places:searchText'
    #headers used for the request
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': GOOGLE_API_KEY,
        'X-Goog-FieldMask': '*'
    }
    #the request body used for the api request
    request_body = {
        "textQuery": query,
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": 100.0
            }
        }
    }
    #get the response
    response = requests.post(url, headers=headers, json=request_body)
    #get the json object
    output = response.json()
    #save the locations in a dictionary
    locations = []
    #loop through the output
    for row in output['places']:
        locations.append({
            "name": row["displayName"]["text"],
            "address": row["formattedAddress"],
            #TODO: contact information
        })
    return {
        "query": query,
        "latitude_used": latitude,
        "longitude_used": longitude,
        "results": locations
    }

locater_agent = create_react_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_places],
    prompt=(
        "You are a locater agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with locating-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="locater_agent",
)

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-mini"),
    agents=[research_agent, locater_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent, such as more information on city guidelines.\n"
        "- a locater agent. Assign locating-related tasks to this agent, such as finding places near a specific area.\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "You should use the research agent to inform yourself on the appropriate guidelines and then use the locater agent to give five locations for the user.\n"
        "You must also inform the user of any fines they could incur if they do not follow the guidelines.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hello, somebody ran over a racoon and left it on the street, what should I do? By the way I live in Koreatown, Los Angeles.",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]