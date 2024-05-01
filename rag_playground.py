from dotenv import load_dotenv, find_dotenv
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms.openai import OpenAI
import logging
import openai
import os
import streamlit as st
import sys

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OpenAI_API_Key')

def initialize_app():

    global default_prompt_template
    global callback_manager
    global memory
    global llm_choices

    default_prompt_template = ("Context information has been provided below:\n"
                               "---------------------\n"
                               "{context}\n"
                               "---------------------\n"
                               "Given this context information and not prior knowledge, "
                               "answer the query in a friendly and helpful manner.\n"
                               "Query: {query}\n"
                               "Answer: ")

    llama_debug = LlamaDebugHandler(print_trace_on_end = True)
    callback_manager = CallbackManager([llama_debug])

    memory = ChatMemoryBuffer.from_defaults(token_limit = 1500)

    llm_choices = {'https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf': 'Llama-2-13B-Chat-GGUF', 
                   'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf': 'Llama-2-7B-Chat-GGUF',
                   'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf': 'Mistral-7B-Instruct-V0.2-GGUF',
                   'https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf': 'Mixtral-8X7B-Instruct-V0.1-GGUF',
                   'OpenAI-GPT-3.5-Turbo-0125': 'OpenAI-GPT-3.5-Turbo-0125'}
    
def initialize_page():

    st.set_page_config(page_title = 'Andalem RAG Playground', 
                       page_icon = './app_images/andalem-icon.png', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')

    custom_style = """
                        <style>

                            footer {visibility: hidden;}
                            header {visibility: visible;}
                            #MainMenu {visibility: hidden;}

                            [data-testid=ScrollToBottomContainer] {
                                margin-top: -20px;
                            }

                            [data-testid=stSidebarUserContent] {
                                margin-top: -50px;
                            }
                            
                            [data-testid=stImage] {                                
                                text-align: center;
                                display: block;
                                margin-left: auto;
                                margin-right: auto;
                                width: 100%;
                            }

                            [data-testid=StyledFullScreenButton] {
                                display: none;
                            }

                        </style>
                   """
    st.markdown(custom_style, unsafe_allow_html = True)

    global chosen_llm
    global temperature
    global top_p
    global top_k
    global role_toggle
    global custom_prompt_template

    st.image('./app_images/andalem-logo.png', width = 245)

    with st.expander('Upload PDFs'):

        if 'pdf_uploader_key' not in st.session_state:
            
            st.session_state['pdf_uploader_key'] = 0

        pdf_uploader = st.file_uploader('Upload PDFs', type = 'pdf', accept_multiple_files = True, label_visibility = 'hidden', key = st.session_state['pdf_uploader_key'])

        upload_pdfs_button = st.button('Upload', key = 'upload_pdfs_button')

        if pdf_uploader is not None:

            if upload_pdfs_button:

                upload_pdfs(pdf_uploader)

    with st.sidebar:

        st.header('Settings')

        with st.expander('LLM Settings', expanded = True):

            chosen_llm = st.selectbox('Choose LLM:', options = list(llm_choices.keys()), index = 4, format_func = chosen_llm_path_or_url, key = 'chosen_llm')

            temperature = st.slider('Temperature:', min_value = 0.0,
                                                    max_value = 1.0, 
                                                    value = 0.05, 
                                                    step = 0.01)
        
            top_p = st.slider('Top P:', min_value = 0.0,
                                        max_value = 1.0, 
                                        value = 0.02, 
                                        step = 0.01)
            
            top_k = st.slider('Top K:', min_value = 1,
                                        max_value = 50, 
                                        value = 2, 
                                        step = 1)
        
        with st.expander('Chat Settings', expanded = False):

            role_toggle = st.toggle('Submit as Assistant', value = False, key = 'role_toggle')
                
            custom_prompt_template = st.text_area('Prompt Template:', default_prompt_template, height = 200, key = 'custom_prompt_template')

    if 'messages' not in st.session_state.keys():

        st.session_state.messages = [{'role': 'assistant',
                                      'content': 'How may I be of assistance?'}]  
        
@st.cache_resource(show_spinner = False)
def load_chosen_llm():

    if chosen_llm != 'OpenAI-GPT-3.5-Turbo-0125':

        llm = LlamaCPP(model_path = chosen_llm,
                       temperature = temperature,
                       max_new_tokens = 1024,
                       context_window = 3900,
                       generate_kwargs = {},
                       model_kwargs = {'n_gpu_layers': 0,
                                       'top_p': top_p,
                                       'top_k': top_k},
                       messages_to_prompt = messages_to_prompt,
                       completion_to_prompt = completion_to_prompt,
                       verbose = True)
        
    else:

        pass    

    if chosen_llm == 'OpenAI-GPT-3.5-Turbo-0125':

        with st.spinner(text = 'Loading and indexing the training documents. This will take some time . . .'):
    
            directory_reader = SimpleDirectoryReader(input_dir = './custom_knowledge_documents', recursive = True)

            custom_knowledge_documents = directory_reader.load_data()

            Settings.llm = OpenAI(model = 'gpt-3.5-turbo-0125',
                                  temperature = temperature,
                                  max_new_tokens = 1024,
                                  context_window = 3900,
                                  top_p = top_p,
                                  top_k = top_k,
                                  messages_to_prompt = messages_to_prompt,
                                  completion_to_prompt = completion_to_prompt)
            
            Settings.chunk_size = 100
            Settings.chunk_overlap = 10
            Settings.callback_manager = callback_manager

    else:

        with st.spinner(text = 'Loading and indexing the training documents. This will take some time . . .'):
    
            directory_reader = SimpleDirectoryReader(input_dir = './custom_knowledge_documents', recursive = True)

            custom_knowledge_documents = directory_reader.load_data()

            Settings.llm = llm
            Settings.chunk_size = 100
            Settings.chunk_overlap = 10
            Settings.embed_model = 'local'
            Settings.callback_manager = callback_manager
        
    index = VectorStoreIndex.from_documents(custom_knowledge_documents, show_progress = True)

    index.storage_context.persist(persist_dir = './vector_database')

    return index         
        
def initialize_messages():

    index = load_chosen_llm()

    if custom_prompt_template == '':

        text_qa_template_string = default_prompt_template

    else:

        text_qa_template_string = custom_prompt_template

    text_qa_template = PromptTemplate(text_qa_template_string)

    if 'query_engine' not in st.session_state.keys():
        
        st.session_state.query_engine = index.as_query_engine(text_qa_template = text_qa_template,
                                                              kwargs = {'chat_mode': 'condense_question',
                                                                        'memory': memory,
                                                                        'verbose': True,
                                                                        'streaming': True})

    if role_toggle:

        if prompt := st.chat_input('Your correction . . .'):

            st.session_state.messages.append({'role': 'assistant', 'content': prompt})

    else:

        if prompt := st.chat_input('Your question . . .'):

            st.session_state.messages.append({'role': 'user', 'content': prompt})        

    for message in st.session_state.messages:

        with st.chat_message(message['role']):

            st.write(message['content'])

    if st.session_state.messages[-1]["role"] != "assistant":

        with st.chat_message('assistant'):

            with st.spinner('Thinking . . .'):

                response_stream = st.session_state.query_engine.query(prompt)

                st.write(response_stream.response)

                message = {'role': 'assistant',
                           'content': response_stream.response}
                
                st.session_state.messages.append(message)          

def upload_pdfs(pdf_uploader):

    for pdf_file in pdf_uploader:

        with open(os.path.join('training_documents', pdf_file.name), 'wb') as file:

            file.write((pdf_file).getbuffer())

    st.session_state['pdf_uploader_key'] += 1

    st.toast('Upload Successful')

    st.cache_data.clear()
    st.cache_resource.clear()  

    st.experimental_rerun()                

def chosen_llm_path_or_url(chosen_llm):

    return llm_choices[chosen_llm]                     
            
if __name__ == '__main__':

    initialize_app()
 
    initialize_page()

    initialize_messages()