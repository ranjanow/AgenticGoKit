# Week 3: The Anatomy of an Agent

## The Three Pillars of AgenticGoKit

### Pillar 1: Multi-Agent Workflows
Multiple specialized agents communicate and hand off tasks through structured pipelines.
AgenticGoKit supports two routing patterns:
- Sequential: Each agent waits for the previous agent's output (step-by-step reasoning)
- Parallel: Multiple agents process distinct tasks at the same time (powered by Go goroutines)

### Pillar 2: Memory and Context Retention
Without memory, an AI starts every interaction with complete amnesia.
AgenticGoKit provides two types of memory:
- Short-term: LLM context window — single session, volatile
- Long-term: Vector database (chromem/pgvector) — persistent across restarts, used for RAG

### Pillar 3: Tools and External Action
An LLM trapped in a text box can only generate words.
Tools allow agents to take real action: web search, SQL queries, file reading, API calls.
AgenticGoKit uses the Model Context Protocol (MCP) as a universal adapter for dynamic tool discovery.

## The Request Lifecycle
1. Input — User sends a prompt
2. Processing — Agent checks Memory and evaluates Workflow routing rules
3. Tool Use — Agent triggers MCP Tools to pull live external data if required
4. Output — Agent synthesizes data and streams final response back to user

## Examples I Ran
- ollama-short-answer: Ran short answer queries using gemma3:1b model via Ollama
- story-writer-chat-v2: Ran a multi-turn story writing agent using Ollama locally

## What I Learned
- A basic LLM is isolated, single-threaded, and forgetful
- AgenticGoKit wraps an LLM in three infrastructural pillars to make it truly autonomous
- Removing any one pillar collapses the system's ability to execute complex tasks reliably
- AGK_TRACE=true enables full tracing of the reasoning lifecycle for debugging