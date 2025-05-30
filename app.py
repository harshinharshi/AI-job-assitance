"""
AI Job Assistance - Streamlit Web Application
============================================

This is the main user interface for the AI Job Assistance system.
It provides a chat-based interface where users can:
1. Upload their CV/resume
2. Interact with the multi-agent system
3. Get job recommendations and cover letters

The app uses Streamlit for the web interface and integrates with
the LangGraph multi-agent system for processing user requests.
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import core system components
from agents import define_graph
from streamlit_chat import message
from llms import load_llm 
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit_pills import pills


# ==============================================
# APPLICATION CONFIGURATION
# ==============================================

# Configure Streamlit page settings
st.set_page_config(
    layout="wide",
    page_title="GenAI Job Agent",
    page_icon="🦜"
)

# Main application title
st.title("GenAI Job Agent - 🦜")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Your CV", 
    type="pdf",
    help="Upload your resume/CV in PDF format for analysis"
)


# ==============================================
# SYSTEM INITIALIZATION
# ==============================================

# Initialize LLM and agent system
llm_name = os.environ.get('LLM_NAME')
llm = load_llm(llm_name)
print(f"Initialized LLM: {llm}")

# Create Streamlit callback handler for displaying agent interactions
st_callback = StreamlitCallbackHandler(st.container())

# Initialize the multi-agent graph
graph = define_graph(llm, llm_name)


# ==============================================
# FILE UPLOAD HANDLING
# ==============================================

if uploaded_file is not None:
    # Create temporary directory for file storage
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Created temporary directory: {temp_dir}")

    # Save uploaded file with standardized name
    bytes_data = uploaded_file.getvalue()
    predefined_name = "cv.pdf"
    file_path = os.path.join(temp_dir, predefined_name)
    
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    
    print(f"CV uploaded successfully: {file_path}")

    # ==============================================
    # CHAT FUNCTIONALITY
    # ==============================================

    def conversational_chat(query: str, graph) -> str:
        """
        Main conversation handler that processes user queries through the agent system.
        
        This function:
        1. Streams the multi-agent execution process
        2. Displays intermediate results from each agent
        3. Stores conversation history
        4. Returns the final combined response
        
        Parameters:
        -----------
        query : str
            User's input query or question
        graph : StateGraph
            Compiled LangGraph workflow for agent execution
            
        Returns:
        --------
        str
            Combined response from all agents that participated in the conversation
        """
        results = []
        
        print(f"Processing query: {query}")
        
        # Stream the graph execution with recursion limit
        for s in graph.stream(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 100},
        ):
            # Skip the final end state
            if "__end__" not in s:
                result = list(s.values())[0]
                
                # Handle agent message outputs
                if 'messages' in result:
                    for message_data in result['messages']:
                        agent_name = message_data.name or 'System'
                        message_content = message_data.content
                        
                        # Store result for history
                        agent_response = f"{agent_name} Agent: {message_content}"
                        results.append(agent_response)
                        
                        # Display in Streamlit interface
                        st.header(f"{agent_name} Agent:")
                        st.write(message_content)
                        
                # Handle routing decisions
                elif 'next' in result:
                    st.write(f"Supervisor Decision: {result}")
                    print(f"Routing to: {result}")

        # Store conversation in session state
        st.session_state['history'].append((query, results))
        
        return ' '.join(results)

    # ==============================================
    # SESSION STATE INITIALIZATION
    # ==============================================

    # Initialize session state variables for UI persistence
    if 'selected_index' not in st.session_state:
        st.session_state['selected_index'] = None
        
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask anything to your Job agent: 🤗"]
        
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    # ==============================================
    # USER INTERFACE LAYOUT
    # ==============================================

    # Create main containers for chat interface
    response_container = st.container()
    container = st.container()

    # User input section
    with container:
        # Pre-defined query options for quick start
        options = [
            "Extract and summarize my CV",
            "Find me Data scientist job in india", 
            "Generate a cover letter for my cv"
        ]
        
        # Pills interface for quick query selection
        selected = pills(
            "Choose a question to get started or write your own below.",
            options,
            clearable=None,
            index=st.session_state['selected_index'],
            key="pills"
        )
        
        # Update selected index when user clicks a pill
        if selected:
            st.session_state['selected_index'] = options.index(selected)

        # Main input form
        with st.form(key='my_form', clear_on_submit=True):
            # Text input with default value from selected pill or previous input
            user_input = st.text_input(
                "Query:", 
                value=(selected if selected else st.session_state.get('input_text', '')), 
                placeholder="Write your query 👉 (:", 
                key='input'
            )
            
            # Submit button
            submit_button = st.form_submit_button(label='Send')

        # ==============================================
        # QUERY PROCESSING
        # ==============================================

        # Process user input when form is submitted
        if submit_button and user_input:
            print(f"User submitted: {user_input}")
            
            # Process query through agent system
            output = conversational_chat(user_input, graph)
            
            # Update conversation history
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['input_text'] = user_input
            st.session_state['selected_index'] = None  # Reset pill selection

    # ==============================================
    # CHAT HISTORY DISPLAY
    # ==============================================

    # Display conversation history
    if st.session_state['generated']:
        with response_container:
            # Show all previous exchanges
            for i in range(len(st.session_state['generated'])):
                # User message
                message(
                    st.session_state["past"][i], 
                    is_user=True, 
                    key=str(i) + '_user', 
                    avatar_style="big-smile"
                )
                
                # Agent response
                message(
                    st.session_state["generated"][i], 
                    key=str(i), 
                    avatar_style="thumbs"
                )

else:
    # ==============================================
    # NO FILE UPLOADED STATE
    # ==============================================
    
    # Show instructions when no CV is uploaded
    st.info("👆 Please upload your CV in the sidebar to get started!")
    st.markdown("""
    ### How to use this AI Job Agent:
    
    1. **Upload your CV/Resume** - Use the file uploader in the sidebar
    2. **Choose or type your query** - Select from quick options or write your own
    3. **Get AI assistance** - The system will analyze your CV, search for jobs, and generate cover letters
    
    ### Available Features:
    - 📊 **CV Analysis**: Extract and summarize your qualifications
    - 🔍 **Job Search**: Find relevant positions based on your profile  
    - ✍️ **Cover Letter Generation**: Create personalized cover letters
    
    ### Supported Queries:
    - "Extract and summarize my CV"
    - "Find me [job title] jobs in [location]"
    - "Generate a cover letter for [specific job]"
    - Custom queries about job search and career assistance
    """)


