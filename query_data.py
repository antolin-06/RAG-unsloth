import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import get_embedding_function

CHROMA_PATH = "chroma"
API_KEY_GOOGLE = "AIzaSyATzPGkB_68CC2zZzOAgUVnWio7nGyNRhk"
PROMPT_TEMPLATE = """
Responde a la pregunta basándote únicamente en la siguiente información:

{context}

---

Responde a la siguiente pregunta basándote en la información proporcionada arriba: {question}
"""

def query_rag(query_text: str):

    vector_store = Chroma(
        collection_name="instrucciones_google",  # Nombre de la colección
        embedding_function=get_embedding_function.get_embedding_function_google(API_KEY_GOOGLE),  # Función de embedding
        persist_directory=CHROMA_PATH,  # Directorio en el que queremos que se guarde
    )

    ### BUILDING PROMPT
    #Search in db
    results = vector_store.similarity_search_with_score(query_text, k=3) # Here we are selecting the top 5 similarity chunks based
    # on the comparison we are selecting by the function we are calling. Other functions are: similarity_search_with_vectors,
    # _select_relevance_score_fn, max_marginal_relevance_search_by_vector ...


    #Giving format to final prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    ### CALLING LLM AND CREATING FINAL RESPONSE
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)