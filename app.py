import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from agents import define_graph
from streamlit_chat import message
from llms import load_llm 
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit_pills import pills

st.set_page_config(layout="wide")
st.title("GenAI Job Agent - 🦜")
uploaded_file = st.sidebar.file_uploader("Upload Your CV", type="pdf")

llm_name=os.environ.get('LLM_NAME')
llm = load_llm(llm_name)
print(llm)
st_callback = StreamlitCallbackHandler(st.container())
graph = define_graph(llm, llm_name)

# Handle file upload
if uploaded_file is not None:
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    bytes_data = uploaded_file.getvalue()
    predefined_name = "cv.pdf"
    
    # To save the file, we use the 'with' statement to open a file and write the contents
    # The file will be saved with the predefined name in the /temp folder
    file_path = os.path.join(temp_dir, predefined_name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)

    #@traceable # Auto-trace this function
    def conversational_chat(query, graph):
        results = []
        #container = st.container(border=True)
        for s in graph.stream(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 100},):
            if "__end__" not in s:
                result = list(s.values())[0]
                if 'messages' in result:
                    for message_data  in result['messages']:
                        name = message_data.name
                        message = message_data.content
                        if name is None:
                            name = ''
                        results.append(name+" Agent: "+message)
                        st.header(name+" Agent: ")
                        st.write(message)
                elif 'next' in result:
                    st.write(result)

        st.session_state['history'].append((query, results))
        return ' '.join(results)

    if 'selected_index' not in st.session_state:
        st.session_state['selected_index'] = None 
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask anything to your Job agent: 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        options =  [
                    "Extract and summarize my CV",
                    "Find me Data scientist job in india",
                    "Generate a cover letter for my cv"               
                    ]
        selected = pills(
                "Choose a question to get started or write your own below.",
                options,
                clearable=None, # type: ignore
                index=st.session_state['selected_index'],
                key="pills"
            )
        if selected:
            st.session_state['selected_index'] = options.index(selected)


        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", value=(selected if selected else st.session_state.get('input_text', '')), placeholder="Write your query 👉 (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            

        if submit_button and user_input:
            output = conversational_chat(user_input, graph)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['input_text'] = user_input  # Save the last input
            st.session_state['selected_index'] = None 

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        