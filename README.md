# mcp_demo_gemini

This is a standard MCP demo from the Anthropic course, modified to use Gemini as the large language model. Learning project for the beginers.

The origin example is with Claude, provided on deeplearning.ai: https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/hg6oi/chatbot-example

## Start MCP Server:
Navigate to the project directory and initiate it with uv:
```bash
cd <project>
uv init
```
Create virtual environment and activate it:
```bash
uv venv
source .venv/bin/activate
```
Install dependencies:
```bash
uv add mcp arxiv
Launch the inspector:
npx @modelcontextprotocol/inspector uv run research_server.py
```
If you get a message asking "need to install the following packages", type: y