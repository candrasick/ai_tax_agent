#!/usr/bin/env python
"""Module for creating reusable LangChain agents for tax analysis."""

import logging

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel

from ai_tax_agent.settings import settings
from ai_tax_agent.llm_utils import get_gemini_llm
from ai_tax_agent.tools.db_tools import get_section_details_and_stats
from ai_tax_agent.tools.chroma_tools import query_cbo_projections, query_form_instructions

logger = logging.getLogger(__name__)

# --- Placeholder Tool Functions ---
# These will be replaced with actual implementations later

def db_query_placeholder(query: str) -> str:
    """Placeholder for querying the relational database."""
    logger.info(f"DB Placeholder received query: {query}")
    return "Placeholder: Database query results for '{query}'. Replace with actual DB query tool."

# --- Agent Creation ---

def create_tax_analysis_agent(
    llm_model_name: str = "gemini-1.5-flash-latest",
    temperature: float = 0.1,
    max_iterations: int = 15,
    verbose: bool = True,
) -> AgentExecutor | None:
    """Creates a LangChain ReAct agent equipped with tools for tax analysis.

    Args:
        llm_model_name: The name of the Gemini model to use.
        temperature: The sampling temperature for the LLM.
        max_iterations: The maximum number of steps the agent can take.
        verbose: Whether the agent executor should run in verbose mode.

    Returns:
        An initialized AgentExecutor instance or None if setup fails.
    """
    logger.info(f"Creating tax analysis agent with model: {llm_model_name}")

    # 1. Initialize LLM
    llm: BaseChatModel | None = get_gemini_llm(
        model_name=llm_model_name, temperature=temperature
    )
    if not llm:
        logger.error("Failed to initialize LLM. Cannot create agent.")
        return None

    # 2. Initialize Tools
    tools = []

    # Calculator Tool
    try:
        # Load tools handles the LLMMathChain setup internally
        calculator_tools = load_tools(["llm-math"], llm=llm)
        tools.extend(calculator_tools)
        logger.info("Calculator tool initialized.")
    except Exception as e:
        logger.warning(f"Could not initialize calculator tool: {e}. Proceeding without it.")

    # SerpAPI Tool
    if settings.serp_api_key:
        try:
            search = SerpAPIWrapper(serpapi_api_key=settings.serp_api_key)
            tools.append(
                Tool(
                    name="Web Search",
                    func=search.run,
                    description="Useful for when you need to answer questions about current events, data, or information not found in internal databases. Use specific search queries.",
                )
            )
            logger.info("SerpAPI web search tool initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize SerpAPI tool: {e}. Proceeding without it.")
    else:
        logger.warning("SERP_API_KEY not found in settings. Web search tool disabled.")

    # Get Section Details and Statistics Tool
    tools.append(
        Tool.from_function(
            func=get_section_details_and_stats,
            name="Get Section Details and Statistics",
            description="Retrieves aggregated financial statistics (dollars, forms, people) AND a detailed list of linked form fields (including labels, instruction info, individual stats, and text snippets) for a specific US Code section identifier (e.g., '162' or a section ID). Use this to understand the detailed statistical impact and context of a section.",
        )
    )
    logger.info("'Get Section Details and Statistics' database tool initialized.")

    # Query CBO Projections Tool
    tools.append(
        Tool.from_function(
            func=query_cbo_projections,
            name="Query CBO Revenue Projections",
            description="Performs semantic similarity search on the CBO Revenue Projections data. Useful for finding relevant CBO projection details based on a textual query like 'projected corporate income tax 2026' or 'individual income tax trends'.",
        )
    )
    logger.info("'Query CBO Revenue Projections' ChromaDB tool initialized.")

    # Query Form Instructions Tool
    tools.append(
        Tool.from_function(
            func=query_form_instructions,
            name="Query Form Instructions",
            description="Performs semantic similarity search on indexed form field instructions/text. Useful for finding relevant form field details or instructions when direct links via section number are missing or insufficient. Use queries like 'find instructions about deducting business meals' or 'search form text for capital gains reporting'.",
        )
    )
    logger.info("'Query Form Instructions' ChromaDB tool initialized.")

    if not tools:
        logger.error("No tools could be initialized. Cannot create agent.")
        return None

    # 3. Get ReAct Prompt Template
    # Using a standard ReAct prompt from LangChain Hub
    try:
        prompt = hub.pull("hwchase17/react")
        logger.info("Pulled ReAct prompt from LangChain Hub.")
    except Exception as e:
        logger.error(f"Failed to pull prompt from LangChain Hub: {e}. Cannot create agent.")
        return None

    # 4. Create the ReAct Agent
    try:
        agent = create_react_agent(llm, tools, prompt)
        logger.info("ReAct agent created successfully.")
    except Exception as e:
        logger.error(f"Failed to create ReAct agent: {e}. Cannot create agent.")
        return None

    # 5. Create the Agent Executor
    try:
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True, # Handle potential output parsing errors
        )
        logger.info(f"AgentExecutor created. Max iterations: {max_iterations}, Verbose: {verbose}")
        return agent_executor
    except Exception as e:
        logger.error(f"Failed to create AgentExecutor: {e}")
        return None

# Example Usage (Optional - can be removed or put in a separate script)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Testing agent creation...")
    tax_agent_executor = create_tax_analysis_agent(verbose=True)

    if tax_agent_executor:
        logger.info("Agent created successfully. Testing tools...")
        try:
            # Test calculator
            response = tax_agent_executor.invoke({"input": "What is 7% of $550,000?"})
            logger.info(f"\n--- Calculator Response ---\n{response}\n")

            # Test DB tool
            response = tax_agent_executor.invoke({"input": "Get details and statistics for section 162"})
            logger.info(f"\n--- DB Tool Response ---\n{response}\n")

            # Test web search
            response = tax_agent_executor.invoke({"input": "Search web for current US unemployment rate"})
            logger.info(f"\n--- Web Search Response ---\n{response}\n")

            # Test Chroma CBO tool
            response = tax_agent_executor.invoke({"input": "Query CBO projections for payroll taxes in 2025"})
            logger.info(f"\n--- Chroma CBO Response ---\n{response}\n")

            # Test Chroma Instructions tool
            response = tax_agent_executor.invoke({"input": "Query form instructions for home office deduction"})
            logger.info(f"\n--- Chroma Instructions Response ---\n{response}\n")

        except Exception as e:
            logger.error(f"Error during agent invocation test: {e}", exc_info=True)
    else:
        logger.error("Agent creation failed.") 