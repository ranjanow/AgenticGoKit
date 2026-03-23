\# My AgenticGoKit Journey



\## Week 1: The Setup \& First Agent



\### What I did

\- Installed Go programming language

\- Installed the AGK CLI tool (v0.2.2)

\- Scaffolded my first AI agent using: agk init my-agent --template quickstart

\- Configured the agent to use Ollama (local LLM) with llama3.2 model

\- Successfully ran the agent and got a haiku response about coding



\### What I learned

\- Go uses go.mod to manage dependencies (similar to package.json)

\- The go mod tidy command downloads all required packages

\- AGK CLI scaffolds a complete agent project structure automatically

\- Ollama runs LLMs locally without needing an API key

\- An AI agent in Go is just a standard Go program using the AgenticGoKit framework



\### Challenges I faced

\- The agent timed out at 30 seconds before Ollama could respond

\- Fixed by increasing the timeout to 120 seconds in main.go



\### My first agent output

User: Write a haiku about coding.

