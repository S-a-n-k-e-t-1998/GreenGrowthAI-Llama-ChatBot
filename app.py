from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import chainlit as cl


from params import CHROMA_DB_SETTINGS,EMBEDDING_MODEL_NAME,PERSIST_DIR,LLM_MODEL_PATH

from prompt import set_custom_prompt

# loading the model
def load_model():
    # load the model 
    llm=CTransformers(
        model=LLM_MODEL_PATH,
        model_type='llama',
        max_new_tokens=512,
        temperature=0.1
    )
    return llm

# Retrieval QA Chain 
def retrieval_qa_chain(llm,prompt,db):
    qa_chain=RetrievalQA.from_chain_type(llm=llm,
                                         chain_type='stuff',
                                         retriever=db.as_retriever(search_kwargs={'k':2}),
                                         return_source_documents=True,
                                         chain_type_kwargs={'prompt':prompt}
                                         )
    return qa_chain 

# QA Model Function
def qa_bot():
    # embedding model
    embeddings=HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device':'cpu'})
    # load the chroma db embedding 
    db=Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        client_settings=CHROMA_DB_SETTINGS) 
    
    # load the llm model
    llm=load_model()
    # call custom prompt
    prompt=set_custom_prompt()
    # Retrieval QA Chain 
    qa_chain=retrieval_qa_chain(llm,prompt,db)

    return qa_chain

# final response
def final_response(query):
    qa_results=qa_bot()
    response=qa_results({'query':query})
    return response
# print(final_response("what is organic farming?"))

#------------------------------------Chainlit Code--------------------------------------------
@cl.on_chat_start
async def start():
    chain=qa_bot()
    msg=cl.Message(content="Starting the bot...") 
    await msg.send()
    msg.content="Hi, Welcome to GreenGrowthAI-Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain",chain)


@cl.on_message
async def main(message):
    chain=cl.user_session.get("chain")

    cb=cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res=await chain.acall(message,callbacks=[cb])
    answer=res['result']
    sources=res['source_documents']

    if sources:
        answer +=  f'\nSources:'+str(sources)
    else:
        answer += '\nNo sources found'

    await cl.Message(content=answer).send()

