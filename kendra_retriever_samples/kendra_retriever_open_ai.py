from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.prompts import PromptTemplate
import os


def build_chain():
    """
    Build the RetrievalQA chain using the configured language model and retriever.
    """
    region = os.environ.get("AWS_REGION")
    kendra_index_id = os.environ.get("KENDRA_INDEX_ID")

    if not region or not kendra_index_id:
        raise ValueError("Please set the AWS_REGION and KENDRA_INDEX_ID environment variables.")

    # Language model configuration
    llm = OpenAI(batch_size=5, temperature=0, max_tokens=300)

    # Configuring the retriever
    retriever = AmazonKendraRetriever(index_id=kendra_index_id)

    # Prompt template for conversational QA with bot introduction
    prompt_template = """
    Hi, I'm a helpful bot with access to over 100k documents of data from COVID/mental health resources.
    These documents are sourced from data.gov, a website that provides access to a wide range of open government data.
    Feel free to ask me anything related to COVID or mental health, and I'll do my best to provide you with accurate information.
    The following is a friendly conversation between a human and me, the bot. 
    I provide lots of specific details from my context.
    If I don't know the answer to a question, I truthfully say I do not know.

    {context}

    Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
    if not present in the document. 
    Solution:"""

    # Creating a prompt template object
    prompt_template_obj = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Chain type kwargs
    chain_type_kwargs = {"prompt": prompt_template_obj}

    return RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )


def run_chain(chain, prompt: str, history=[]):
    """
    Run the retrieval QA chain with the given prompt and return the result.

    Args:
        chain (RetrievalQA): The retrieval QA chain.
        prompt (str): The user's query.
        history (list, optional): A list of previous conversation history. Defaults to an empty list.

    Returns:
        dict: A dictionary containing the answer and source documents.
    """
    # Run the chain
    result = chain(prompt, history=history)

    # To make it compatible with chat samples
    return {
        "answer": result['result'],
        "source_documents": result['source_documents']
    }


if __name__ == "__main__":
    try:
        chain = build_chain()
    except ValueError as e:
        print(f"Error: {e}")
    else:
        prompt = input("Enter your question: ")
        if prompt.strip():
            result = run_chain(chain, prompt)
            print(f"Answer: {result['answer']}")
            if 'source_documents' in result:
                print('Sources:')
                for document in result['source_documents']:
                    print(document.metadata['source'])
        else:
            print("Please provide a non-empty prompt.")
