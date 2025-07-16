import arxiv
import json
import os
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

PAPER_DIR = "papers"

def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

# search_papers("computers")
# print(extract_info('1310.7911v2'))

tools = [
    {
        "name": "search_papers",
        "description": "Search for papers on arXiv based on a topic and store their information.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to search for"
                }, 
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to retrieve",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "extract_info",
        "description": "Search for information about a specific paper across all topic directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The ID of the paper to look for"
                }
            },
            "required": ["paper_id"]
        }
    }
]

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # For any other type, convert using str()
        result = str(result)
    return result

def process_query(query):

    contents = [
        types.Content(
            role="user", parts=[types.Part(text=query)]
        )
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )
    process_query = True

    while process_query:
        assistant_content = []
        function_call = response.candidates[0].content.parts[0].function_call
        if function_call:
            contents.append(types.Content(role="model", parts=[types.Part(text=str(assistant_content))]))
            
            print(f"Function to call: {function_call.name} with arguments: {function_call.args}")
            result = execute_tool(function_call.name, function_call.args)
            # Create a function response part
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={"result": result},
            )
            # Append function call and result of the function execution to contents
            contents.append(response.candidates[0].content) # Append the content from the model's response.
            contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                config=config,
                contents=contents,
            )
            if len(response.candidates[0].content.parts) == 1 and response.candidates[0].content.parts[0].text:
                process_query = False
                print(f"Final response: {response.candidates[0].content.parts[0].text}") 
        else:
            candidate = response.candidates[0]
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                text_response = ' '.join(text_parts)
                print(text_response)
                assistant_content.append(text_response)
            else:
                print("No response generated")
            if len(candidate.content.parts) == 1:
                process_query = False

    

def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")

load_dotenv() 
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()
tool_types = types.Tool(function_declarations=tools)
config = types.GenerateContentConfig(tools=[tool_types])

chat_loop()

