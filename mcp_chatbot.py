from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from typing import List, Dict, TypedDict
from google import genai
from google.genai import types
from contextlib import AsyncExitStack
import json
import asyncio

PAPER_DIR = "papers"

load_dotenv()
chat_history_limit = 100

class MCP_Chatbot:
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.tool_types = None
        self.config = None
        self.model = "gemini-2.5-flash"
        self.contents = []
        self.available_tools: List[types.FunctionDeclaration] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    def clean_schema(self, obj):
        """递归清理 schema 中不兼容的字段"""
        if isinstance(obj, dict):
            # 移除 Gemini Schema 不支持的字段
            forbidden_fields = {
                '$schema', '$id', 'exclusiveMaximum', 'exclusiveMinimum',
                'const', 'examples', 'if', 'then', 'else', 'allOf', 
                'anyOf', 'oneOf', 'not', 'dependencies', 'patternProperties',
                'additionalItems', 'contains', 'propertyNames',
                'additional_properties', 'additionalProperties'
            }
            
            cleaned = {}
            for key, value in obj.items():
                if key not in forbidden_fields:
                    # 特殊处理 format 字段
                    if key == 'format' and value not in ['enum', 'date-time']:
                        continue  # 跳过不支持的 format 值
                    cleaned[key] = self.clean_schema(value)
            return cleaned
        elif isinstance(obj, list):
            return [self.clean_schema(item) for item in obj]
        else:
            return obj

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()

            # collect tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to the tool server {server_name} with tools:", [tool.name for tool in tools])
            for tool in tools: 
                self.tool_to_session[tool.name] = session
                cleaned_schema = self.clean_schema(dict(tool.inputSchema))
                gemini_tool = types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=types.Schema(**cleaned_schema) if isinstance(cleaned_schema, dict) else None
                )
                self.available_tools.append(gemini_tool)

            # collect resources
            try:
                resourece_response = await session.list_resources()
                if resourece_response and resourece_response.resources:
                    resources = resourece_response.resources    
                    for resource in resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
                        print(f"Connected to the resource server {server_name} with resource: {resource_uri}") 
            except McpError as e:
                if "Method not found" in str(e):
                    print(f"Server {server_name} does not support list_resources(), skipping...")
                else:
                    raise
        except Exception as e:
            print(f"Error connecting to server {server_name}: {e}")
            raise

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers",{})
            tasks = []
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
            if self.available_tools:
                self.tool_types = types.Tool(function_declarations=self.available_tools)
                self.config = types.GenerateContentConfig(tools=[self.tool_types])
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def cleanup(self):
        """Clean up all MCP sessions."""
        await self.exit_stack.aclose()

    def append_content(self, content):
        if len(self.contents) >= chat_history_limit:
            self.contents.pop(0)
        self.contents.append(content)

    async def process_query(self, query):
        self.append_content(types.Content(
            role="user", parts=[types.Part(text=query)]
        ))

        response = self.client.models.generate_content(
            model= self.model,
            contents=self.contents,
            config=self.config,
        )
        process_query = True

        while process_query:
            assistant_content = []
            if not response.candidates or not response.candidates[0].content:
                print("No response generated")
                break
                
            candidate = response.candidates[0]
            content = candidate.content
            if not content or not content.parts:
                print("No content parts in response")
                break
            
            # 处理所有函数调用
            function_calls = []
            for part in content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "args": part.function_call.args
                    })
            
            if function_calls:
                # 直接处理所有函数调用
                results = []
                for call in function_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    print(f"Function to call: {tool_name} with arguments: {tool_args}")

                    session = self.tool_to_session.get(tool_name)
                    if session and tool_name:
                        result = await session.call_tool(tool_name, tool_args)
                        results.append({
                            "name": tool_name,
                            "response": {"result": result}
                        })
                    else:
                        print("Session not available or tool name is None")
                        break
                # 创建单个组合的函数响应
                combined_response = {
                    "tool_responses": results
                }
                function_response_part = types.Part.from_function_response(
                    name="combined_tool_response",
                    response=combined_response,
                )
                # Append function call and result of the function execution to contents
                if content:
                    self.append_content(content) # Append the content from the model's response.
                self.append_content(types.Content(
                    role="user", 
                    parts=[function_response_part]
                )) # Append the function response
                response = self.client.models.generate_content(
                    model=self.model,
                    config=self.config,
                    contents=self.contents,
                )
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts and len(response.candidates[0].content.parts) == 1 and response.candidates[0].content.parts[0].text:
                    print(f"Final response: {response.candidates[0].content.parts[0].text}") 
                    process_query = False

            else:
                text_parts = [part.text for part in content.parts if hasattr(part, 'text') and part.text]
                if text_parts:
                    text_response = ' '.join(text_parts)
                    print(text_response)
                    assistant_content.append(text_response)
                    self.append_content(types.Content(role="model", parts=[types.Part(text=str(assistant_content))]))
                else:
                    print("No response generated")
                if len(content.parts) == 1:
                    process_query = False

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
        
        # Fallback for papers URIs - try any papers resource session
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break

                if query.startswith('@'):
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
        



async def main():
    chatbot = MCP_Chatbot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

