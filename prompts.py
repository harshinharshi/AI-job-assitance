"""
Prompt templates and configurations for different LLM providers.

This module contains all prompt templates used by the multi-agent system,
with support for different LLM providers (OpenAI, Groq/Llama3).
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Llama3 specific template markers for proper formatting
LLAMA3_BEGIN_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> "
LLAMA3_END_TEMPLATE = " <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"


def routing_prompt(llm_name: str, options: list, members: list) -> ChatPromptTemplate:
    """
    Create routing prompt for supervisor agent to decide next action.
    
    Args:
        llm_name (str): Name of the LLM provider ('openai' or 'groq')
        options (list): Available routing options including agents and 'FINISH'
        members (list): List of available agent members
        
    Returns:
        ChatPromptTemplate: Configured prompt template for routing decisions
    """
    system_prompt = get_system_prompt(llm_name)

    if llm_name == 'openai':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system", 
                "Given the conversation above, who should act next? "
                "Or is the task complete and should we FINISH? Select one of: {options}"
            ),
        ]).partial(options=str(options), members=", ".join(members))
        
    elif llm_name == 'groq':
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system", 
                LLAMA3_BEGIN_TEMPLATE + 
                "Summarize and assess the conversation. Given the conversation above, who should act next? "
                "Or is the task complete and should we FINISH? Select one of: {options}" + 
                LLAMA3_END_TEMPLATE
            ),
        ]).partial(options=str(options), members=", ".join(members))
        
    return prompt


def get_system_prompt(llm_name: str) -> str:
    """
    Get the system prompt for supervisor agent based on LLM provider.
    
    Args:
        llm_name (str): Name of the LLM provider ('openai' or 'groq')
        
    Returns:
        str: System prompt template for supervisor agent
    """
    if llm_name == 'openai':
        return (
            "You are a supervisor agent tasked with managing a conversation between the "
            "following workers: {members}. User has uploaded a document and sent a query. "
            "Given the uploaded document and following user request, respond with the worker to act next. "
            "Each worker will perform a task and respond with their results and status. "
            "Only route the tasks based on the router if there is anything to route or task is not complete. "
            "When finished, respond with FINISH."
        )
    elif llm_name == 'groq':
        return (
            LLAMA3_BEGIN_TEMPLATE + 
            "You are a supervisor agent tasked with managing a conversation between the "
            "following workers: {members}. User has uploaded a CV and sent a query. "
            "Given the uploaded CV and following user request, respond with the worker to act next. "
            "Each worker will perform a task and respond with their results and status. "
            "After the result: ask yourself from the original query if the task is satisfied? "
            "Based on that pass it to next appropriate route. "
            "When task is finished, respond with FINISH." +
            LLAMA3_END_TEMPLATE
        )


def get_search_agent_prompt(llm_name: str) -> str:
    """
    Get the prompt for search agent based on LLM provider.
    
    Args:
        llm_name (str): Name of the LLM provider ('openai' or 'groq')
        
    Returns:
        str: Prompt for search agent
    """
    base_prompt = (
        "Search for job listings based on user-specified parameters, "
        "DISPLAY job title, company URL, location, and a summary. "
        "If unsuccessful, retry with alternative keywords up to three times and provide the results"
    )
    
    if llm_name == 'openai':
        return base_prompt
    elif llm_name == 'groq':
        return (
            LLAMA3_BEGIN_TEMPLATE + 
            "You are a Searcher Agent. " + base_prompt + 
            LLAMA3_END_TEMPLATE
        )


def get_analyzer_agent_prompt(llm_name: str) -> str:
    """
    Get the prompt for analyzer agent based on LLM provider.
    
    Args:
        llm_name (str): Name of the LLM provider ('openai' or 'groq')
        
    Returns:
        str: Prompt for analyzer agent
    """
    base_prompt = (
        "Analyze the content of a user-uploaded document and matching job listings "
        "to recommend the best job fit, detailing the reasons behind the choice."
    )
    
    if llm_name == 'openai':
        return base_prompt
    elif llm_name == 'groq':
        return (
            LLAMA3_BEGIN_TEMPLATE + 
            "You are an Analyzer Agent. "
            "Analyze the content of the user-uploaded CV and matching job listings "
            "to recommend the best job fit, detailing the reasons behind the choice." + 
            LLAMA3_END_TEMPLATE
        )


def get_generator_agent_prompt(llm_name: str) -> str:
    """
    Get the prompt for generator agent based on LLM provider.
    
    Args:
        llm_name (str): Name of the LLM provider ('openai' or 'groq')
        
    Returns:
        str: Prompt for generator agent
    """
    base_prompt = "Generate a personalized cover letter based on an uploaded CV and provide the text output."
    
    if llm_name == 'openai':
        return base_prompt
    elif llm_name == 'groq':
        return (
            LLAMA3_BEGIN_TEMPLATE + 
            "You are a Generator Agent. " + base_prompt + 
            LLAMA3_END_TEMPLATE
        )


# Example usage documentation
"""
Example input query:
Find data science job for me in Germany maximum 5 relevant ones.
Then analyze my CV and write me a cover letter according to the best matching job.
"""