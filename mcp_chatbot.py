from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, Tool, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from typing import List, Dict, TypedDict
from google import genai
from google.genai import types
from contextlib import AsyncExitStack
import json
import asyncio
import getpass
import os
import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

PAPER_DIR = "papers"

load_dotenv()
chat_history_limit = 100

class MCP_Chatbot:
    def __init__(self):
        self.sessions = {}
        self.exit_stack = AsyncExitStack()
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
        self.model_with_tools = None
        self.tool_types = None
        self.config = None
        self.model = "gemini-2.5-flash"
        self.available_tools: List[types.FunctionDeclaration] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.system_message = SystemMessage(content="You help search papers and answer questions about them")
        self.conversation_history = []

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
                # 转换MCP工具为LangChain工具
                langchain_tools = self.convert_mcp_tools_to_langchain()
                self.model_with_tools = self.client.bind_tools(langchain_tools)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    def convert_mcp_tools_to_langchain(self):
        """将MCP工具转换为LangChain工具"""
        langchain_tools = []
        def create_tool_factory(name, desc):
            async def tool_func(**kwargs):
                session = self.tool_to_session.get(name)
                if session:
                    result = await session.call_tool(name, kwargs)
                    return result.content[0].text if result.content else "No result"
                return "Tool not available"
            tool_func.__name__ = name
            tool_func.__doc__ = desc
            return tool(tool_func)

        for mcp_tool in self.available_tools:
            # 为每个MCP工具创建LangChain工具
            curr_name = mcp_tool.name
            curr_desc = mcp_tool.description
            langchain_tool = create_tool_factory(curr_name, curr_desc)
            langchain_tools.append(langchain_tool)
        
        return langchain_tools

    def append_content(self, content, message_type=HumanMessage, tool_call_id=None):
        if len(self.conversation_history) >= chat_history_limit:
            self.conversation_history = self.conversation_history[1:]
        if message_type == ToolMessage:
            self.conversation_history.append(message_type(content=content, tool_call_id=tool_call_id))
        else:
            self.conversation_history.append(message_type(content=content))

    async def process_query(self, query):
        self.append_content(query, HumanMessage)
        process_query = True

        while process_query:
            response = self.model_with_tools.invoke([self.system_message] + self.conversation_history)
            if not hasattr(response, 'id') or not response.id:
                print("No response generated")
                break
            self.append_content(response, AIMessage)
            
            # 处理所有函数调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # 直接处理所有函数调用
                for call in response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    tool_call_id = call["id"]
                    print(f"Function to call: {tool_name} with arguments: {tool_args} and id: {tool_call_id}")

                    session = self.tool_to_session.get(tool_name)
                    if session and tool_name:
                        tool_result = await session.call_tool(tool_name, tool_args)
                        self.append_content(tool_result, ToolMessage, tool_call_id)
                    else:
                        print("Session not available or tool name is None")
                        break
                continue
            # exit and print msg if this is not a tool call
            else:
                print(response.content)
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
    
    async def cleanup(self):
        """Clean up all MCP sessions."""
        await self.exit_stack.aclose()

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit' or query.lower() == 'exit':
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
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    chatbot = MCP_Chatbot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

