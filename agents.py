"""
AI Job Assistance - Multi-Agent System
=====================================

This module defines the core multi-agent architecture using LangGraph and LangChain.
The system consists of:
- Supervisor Agent: Routes tasks between specialized agents
- Analyzer Agent: Analyzes CV content and job matches
- Searcher Agent: Finds relevant job postings
- Generator Agent: Creates personalized cover letters

Architecture Pattern: Multi-Agent Supervisor with Worker Nodes
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
import operator
from typing import Annotated, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
from tools import *
from prompts import *
import os 


class AgentState(TypedDict):
    """
    Defines the state structure shared across all agents in the graph.
    
    Attributes:
        input (str): Original user input/query
        messages (Annotated[Sequence[BaseMessage], operator.add]): 
            Chat history with automatic message accumulation
        next (str): Next agent to execute (determined by supervisor)
    """
    input: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str) -> AgentExecutor:
    """
    Factory function to create a specialized agent with specific tools and prompt.
    
    Args:
        llm (ChatOpenAI): Language model instance
        tools (list): List of tools available to this agent
        system_prompt (str): System prompt defining agent behavior
        
    Returns:
        AgentExecutor: Configured agent executor ready for task execution
    """
    # Create prompt template with system instructions and message placeholders
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create OpenAI tools-compatible agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Wrap agent in executor for execution management
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state: AgentState, agent: AgentExecutor, name: str) -> dict:
    """
    Wrapper function that converts agent executors into graph nodes.
    
    This function:
    1. Executes the agent with current state
    2. Formats the output as a HumanMessage
    3. Returns updated state with agent's response
    
    Args:
        state (AgentState): Current graph state
        agent (AgentExecutor): Agent to execute
        name (str): Agent name for message attribution
        
    Returns:
        dict: Updated state with agent's response message
    """
    print(f"EXECUTING {name.upper()} AGENT WITH STATE:", state)
    
    # Execute agent and get result
    result = agent.invoke(state)
    
    # Return formatted message with agent name attribution
    return {
        "messages": [HumanMessage(content=result["output"], name=name)]
    }


def debug_output(data):
    """Debug utility function for development/troubleshooting."""
    print("DEBUG OUTPUT:", data)
    return data


def flatten_output(data: dict) -> dict:
    """
    Flattens nested 'args' structure from Groq/Llama tool outputs.
    
    This is needed because different LLM providers return tool calls
    in slightly different formats. This normalizes the structure.
    
    Args:
        data (dict): Raw tool output with potential 'args' nesting
        
    Returns:
        dict: Flattened dictionary with args content at top level
    """
    if 'args' in data and isinstance(data['args'], dict):
        # Extract args content and merge with top level
        args_content = data.pop('args')
        data.update(args_content)
    return data


def define_graph(llm: ChatOpenAI, llm_name: str) -> StateGraph:
    """
    Main function that constructs the multi-agent workflow graph.
    
    This creates:
    1. Supervisor agent for routing decisions
    2. Three specialized worker agents (Analyzer, Searcher, Generator)
    3. Workflow graph with proper edges and conditional routing
    
    Args:
        llm (ChatOpenAI): Language model instance
        llm_name (str): LLM provider name ('openai', 'groq', 'llama3')
        
    Returns:
        StateGraph: Compiled workflow graph ready for execution
    """
    
    # ==============================================
    # AGENT CONFIGURATION
    # ==============================================
    
    # Define available worker agents
    members = ["Analyzer", "Generator", "Searcher"]
    
    # Define routing options (workers + finish command)
    options = ["FINISH"] + members
    
    # ==============================================
    # SUPERVISOR SETUP
    # ==============================================
    
    # Define function schema for routing decisions
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                }
            },
            "required": ["next"],
        },
    }

    # Get LLM-specific prompts
    prompt = routing_prompt(llm_name, options, members)
    
    print(f"Configuring supervisor for LLM: {llm_name}")
    
    # Create supervisor chain based on LLM provider
    if llm_name == "openai":
        # OpenAI uses function calling
        supervisor_chain = (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route") 
            | JsonOutputFunctionsParser()
        )
    elif llm_name in ["groq", "llama3"]:
        # Groq/Llama use tool calling with different parsing
        supervisor_chain = (
            prompt
            | llm.bind_tools(tools=[function_def]) 
            | JsonOutputToolsParser(first_tool_only=True)
            | flatten_output  # Normalize output structure
        )
        print("DEBUG: Supervisor chain configured for Groq/Llama")

    # ==============================================
    # WORKER AGENT SETUP
    # ==============================================
    
    # Create Searcher Agent - handles job search operations
    search_agent = create_agent(
        llm=llm, 
        tools=[job_pipeline], 
        system_prompt=get_search_agent_prompt(llm_name)
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Searcher")

    # Create Analyzer Agent - handles CV analysis and job matching
    analyzer_agent = create_agent(
        llm=llm, 
        tools=[extract_cv], 
        system_prompt=get_analyzer_agent_prompt(llm_name)
    )
    analyzer_node = functools.partial(agent_node, agent=analyzer_agent, name="Analyzer")

    # Create Generator Agent - handles cover letter generation
    generator_agent = create_agent(
        llm=llm, 
        tools=[generate_letter_for_specific_job], 
        system_prompt=get_generator_agent_prompt(llm_name)
    )
    generator_node = functools.partial(agent_node, agent=generator_agent, name="Generator")

    # ==============================================
    # GRAPH CONSTRUCTION
    # ==============================================
    
    # Initialize workflow graph with shared state
    workflow = StateGraph(AgentState)
    
    # Add all nodes to the graph
    workflow.add_node("Analyzer", analyzer_node)
    workflow.add_node("Searcher", search_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("supervisor", supervisor_chain)

    # ==============================================
    # EDGE CONFIGURATION
    # ==============================================
    
    # All worker agents report back to supervisor after completion
    for member in members:
        workflow.add_edge(member, "supervisor")
    
    # Configure conditional routing from supervisor
    conditional_map = {k: k for k in members}  # Route to worker agents
    conditional_map["FINISH"] = END            # Route to end state
    
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"],  # Extract 'next' field from supervisor output
        conditional_map
    )
    
    # Set supervisor as entry point
    workflow.set_entry_point("supervisor")

    # Compile and return the executable graph
    graph = workflow.compile()
    return graph


# ==============================================
# WORKFLOW EXECUTION FLOW
# ==============================================
"""
Typical execution flow:

1. START → Supervisor receives user input
2. Supervisor analyzes request and routes to appropriate agent:
   - "Extract CV" → Analyzer Agent
   - "Find jobs" → Searcher Agent  
   - "Generate cover letter" → Generator Agent
3. Selected agent executes task using specialized tools
4. Agent reports results back to Supervisor
5. Supervisor decides: route to another agent OR finish
6. Process continues until Supervisor returns "FINISH"

State Management:
- All agents share the same AgentState
- Messages accumulate automatically (operator.add)
- Supervisor maintains routing decisions in 'next' field
"""