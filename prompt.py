from langchain import PromptTemplate

custom_prompt_template="""Use the following pieces of information to answer the user's question. If you 
don't know the answer, just say you don't know. Don't try to make up answers.
context:{context}
question:{question}
only return answer from above context nothinh else.
answer:
"""
# define the custom prompt for knowledge base qa chatbot
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt=PromptTemplate(template=custom_prompt_template,
                          input_variables=['context','question'])
    return prompt

