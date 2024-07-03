import streamlit as st
import logging
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    AzureMLEndpointApiType,
    LlamaChatContentFormatter,
)
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_ai21.chat_models import ChatAI21
from langchain.chains import LLMChain, SequentialChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere.chat_models import ChatCohere
from src.llm_management.prompts import PROMPTS_TEMPLATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s - %(message)s",
)

load_dotenv()

cols = st.columns(3)
cols[1].image("frontend/src/assets/logo_niji.png")

centered_text = """
    <div style="text-align: center;">
        <h1 style="color:#299ADC">Chat with your LLM</h1>
    </div>
    """
st.markdown(centered_text, unsafe_allow_html=True)

# Define the mapping for temperature
temperature_mapping = {
    "Very Precise": 0,
    "A Bit Precise": 0.2,
    "Moderate": 0.5,
    "Creative": 1,
}

frequency_mapping = {
    "very few repetitions": 2.0,
    "few repetitions": 1.0,
    "default": 0.0,
    "more repetitions": -1.0,
    "much more repetitions": -2.0,
}

llm_propositions = [
    "gpt-4o",
    "gpt-4",
    "gpt-35-turbo",
    "Llama-3-8B-Instruct",
    "Llama-3-70B-Instruct",
    "Mistral Large",
    "Mistral Small",
    "AI21-Jamba-Instruct",
    # "Cohere-command-r-plus",
]

# Default PRE_PROMPT
commun_chat_pre_prompt = """
You will use the chat history to help you answer the user message.
Respond to the user message in a helpful and informative way.
Here the chat history:
{chat_history}

Here the user message:
{user_message}"""

# Input section for user to modify the PRE_PROMPT
with st.expander("Configure Chat Assistant"):
    llm_selected = st.selectbox("Select the LLM model", llm_propositions)
    selected_template_name = st.selectbox(
        "Select a prompt template:", list(PROMPTS_TEMPLATE.keys())
    )
    defaul_prompt_template = (
        PROMPTS_TEMPLATE[selected_template_name] + "\n" + commun_chat_pre_prompt
    )
    pre_prompt = st.text_area(
        "Modify the assistant's behavior prompt here:",
        defaul_prompt_template,
        height=300,
    )
    temperature_label = st.select_slider(
        "Select the creativity level (temperature):",
        options=list(temperature_mapping.keys()),
        value="Very Precise",
    )
    frequency_label = st.select_slider(
        "Select the frequency penalty:",
        options=list(frequency_mapping.keys()),
        value="default",
    )
    nb_max_tokens = st.slider(
        "Select the maximum number of tokens:", min_value=10, max_value=5000, value=1000
    )


temperature = temperature_mapping[temperature_label]
frequency_penalty = frequency_mapping[frequency_label]

if "gpt" in llm_selected:
    llm = AzureChatOpenAI(
        deployment_name=llm_selected,
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        max_tokens=nb_max_tokens,
        streaming=True,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )
elif llm_selected == "Llama-3-8B-Instruct":
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=os.getenv("LLAMA_3_8B_ENDPOINT"),
        endpoint_api_type=AzureMLEndpointApiType.serverless,
        endpoint_api_key=os.getenv("LLAMA_3_8B_API_KEY"),
        content_formatter=LlamaChatContentFormatter(),
        max_tokens=nb_max_tokens,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )
elif llm_selected == "Llama-3-70B-Instruct":
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url=os.getenv("LLAMA_3_70B_ENDPOINT"),
        endpoint_api_type=AzureMLEndpointApiType.serverless,
        endpoint_api_key=os.getenv("LLAMA_3_70B_API_KEY"),
        content_formatter=LlamaChatContentFormatter(),
        max_tokens=nb_max_tokens,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )
elif llm_selected == "Mistral Large":
    llm = ChatMistralAI(
        endpoint=os.getenv("MISTRAL_LARGE_ENDPOINT"),
        mistral_api_key=os.getenv("MISTRAL_LARGE_KEY"),
        max_tokens=nb_max_tokens,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )
elif llm_selected == "Mistral Small":
    llm = ChatMistralAI(
        endpoint=os.getenv("MISTRAL_SMALL_ENDPOINT"),
        mistral_api_key=os.getenv("MISTRAL_SMALL_KEY"),
        max_tokens=nb_max_tokens,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )
elif llm_selected == "AI21-Jamba-Instruct":
    llm = ChatAI21(
        model=os.getenv("AI21_MODEL"),
        api_key=os.getenv("AI21_API_KEY"),
        api_host=os.getenv("AI21_API_HOST"),
        max_tokens=nb_max_tokens,
        temperature=temperature,
        model_kwargs={"frequency_penalty": frequency_penalty},
    )

# elif llm_selected == "Cohere-command-r-plus":
#     llm = ChatCohere(
#         cohere_api_key=os.getenv("COHERE_API_KEY"),
#         base_url=os.getenv("COHERE_BASE_URL"),
#         model=os.getenv("COHERE_MODEL"),
#     )


prompt_template = ChatPromptTemplate.from_template(pre_prompt)
chain = SequentialChain(
    chains=[LLMChain(llm=llm, prompt=prompt_template, output_parser=StrOutputParser())],
    input_variables=["chat_history", "user_message"],
    verbose=False,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    history = "\n".join(
        [
            f"""role: {m["role"]}, content: {m["content"]}"""
            for m in st.session_state.messages
        ]
    )
    data = {"chat_history": history, "user_message": prompt}
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_container = st.empty()  # Container for streaming response
        response = ""
        for response_chunk in chain.stream(data):
            logging.info("response_chunk: %s", response_chunk)
            response_container.markdown(response_chunk["text"])
            response += response_chunk["text"]
    st.session_state.messages.append({"role": "assistant", "content": response})
