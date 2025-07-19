from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types

PAPER_DIR = "papers"

nest_asyncio.apply()
load_dotenv()
# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="uv",  # Executable
    args=["run", "research_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)
chat_history_limit = 100

class MCP_Chatbot:
    def __init__(self):
        self.session: ClientSession | None = None
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.
        self.client = genai.Client()
        self.tool_types = None
        self.config = None
        self.model = "gemini-2.5-flash"
        self.contents = []

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
                    
                    if self.session and tool_name:
                        result = await self.session.call_tool(tool_name, tool_args)
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
    

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
        
                if query.lower() == 'quit':
                    break
                    
                await self.process_query(query)
                print("\n")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

        
    async def connect_to_server_and_run(self):
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                response = await session.list_tools()
                
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                gemini_tools = []
                for tool in response.tools:
                    gemini_tool = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                    gemini_tools.append(gemini_tool)

                # The client gets the API key from the environment variable `GEMINI_API_KEY`.
                self.tool_types = types.Tool(function_declarations=gemini_tools)
                self.config = types.GenerateContentConfig(tools=[self.tool_types])
    
                await self.chat_loop()




async def main():
    chatbot = MCP_Chatbot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())

