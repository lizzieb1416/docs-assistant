from common.helper import load_pdf, text_split, download_hugging_face_embeddings, llm_factory

from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def example1():
    '''Prompt template'''
    from langchain_core.prompts import PromptTemplate

    prompt_template = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}."
    )

    prompt = prompt_template.format(adjective="funny", content="chicken")

    llm=llm_factory("llama2-chat")
    answer = llm.invoke(prompt)

    return answer

def example2():
    '''Prompt template: invoke'''
    from langchain_core.prompts import PromptTemplate

    prompt_template = PromptTemplate.from_template(
        "Tell me a {adjective} joke about {content}."
    )

    prompt_val = prompt_template.invoke({"adjective": "funny", "content": "chicken"})

    print(prompt_val.to_string())
    print(prompt_val.to_messages())


def example3():
    '''Chat prompt template'''

    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ]
    )

    messages = chat_template.format_messages(name="Bob", user_input="What is your name?")

    llm= llm_factory("GPT-4")
    answer = llm.invoke(messages)

    return answer

def example4():
    '''Chat prompt template: Piping formatted messages into LangChain's ChatOpenAI chat model class'''

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI bot. Your name is Bob."},
            {"role": "user", "content": "Hello, how are you doing?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
            {"role": "user", "content": "What is your name?"},
        ]
    )

    return response

def example5():
    '''Message prompt template: passing instance of HumanMessagePromptTemplate'''

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import HumanMessagePromptTemplate

    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant that re-writes the user's text to "
                    "sound more upbeat."
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

    messages = chat_template.format_messages(text="I don't like eating healthy things")

    llm= llm_factory("GPT-4")
    answer = llm.invoke(messages)

    return answer

def example6():
    '''MessagePlaceHolder: gives control of what messages to be rendered during formatting'''
    from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
    )
    from langchain_core.messages import AIMessage, HumanMessage

    human_prompt = "Summarize our conversation so far in {word_count} words."
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder(variable_name="conversation"), human_message_template]
    )

    human_message = HumanMessage(content="What is the best way to learn programming?")
    ai_message = AIMessage(
        content="""\
            1. Choose a programming language: Decide on a programming language that you want to learn.

            2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

            3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
            """
    )

    chat = chat_prompt.format_prompt(
        conversation = [human_message, ai_message], word_count=10
    ).to_messages()

    llm= llm_factory("GPT-4")
    answer = llm.invoke(chat)

    return answer

logging.info(example2())
