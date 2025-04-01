from operator import length_hint

import ollama
from click import prompt
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from sympy import vectorize


import get_embedding_function
from langchain_chroma import Chroma

DATA_PATH = "data"
CHROMA_PATH = "chroma"
API_KEY_GOOGLE = "AIzaSyATzPGkB_68CC2zZzOAgUVnWio7nGyNRhk"


### Cargamos de documentos y creación de los chunks
# Función para cargar los PDFs
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load() # Devuelve una lista de tantos objetos como páginas tengan los documentos.

# Función para dividir los documentos en chunks
def split_documents(documents: list[Document]): # La entrada es una lista de 'Document'
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )
    return text_splitter.split_documents(documents)


### STORE EMBEDDED DATA
# Creating the database
def add_to_chroma(chunks: list[Document]):

    vector_store = Chroma(
        collection_name="instrucciones_google",  # Nombre de la colección
        embedding_function=get_embedding_function.get_embedding_function_google(API_KEY_GOOGLE), # Función de embedding
        persist_directory=CHROMA_PATH, # Directorio en el que queremos que se guarde

        # SE PUEDE DEFINIR EL METODO DE COMPARACIÓN PARA CADA COLECCIÓN AQUÍ????????????????? COSENO, EUCLÍDEO...
    )

    # Creamos los IDs
    chunks_with_ids = calculate_chunk_ids(chunks)   # Añadimos nuestro nuevo campo.

    # Comprobamos los chunks existentes en la VS
    existing_items = vector_store.get(include=[])   # Decimos que nos devuelva de la VS los valores entre paréntesis,
                                                    # en nuestro caso no hay nada pero podría ser 'metadatos', por ej.
                                                    # Al dejarlo vacío nos devuelve solo los ids. (creados arriba)
                                                    # {'ids': ['data/Instrucciones-Bang.pdf:0:0', 'data/Instrucciones-Bang.pdf:0:1', 'data/Instrucciones-Bang.pdf:0:2', ...

    existing_ids = set(existing_items["ids"])   # Convertimos la lista de ids en un conjunto para realizar búsquedas rápidas.
                                                # ['data/Instrucciones-Bang.pdf:0:0', 'data/Instrucciones-Bang.pdf:0:1',...]

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []                                     # Creamos la lista vacía de chunks.
    for chunk in chunks_with_ids:                       # Recorremos la lista de chunks y comprobamo si existen o no.
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)                    # Si no existen los añadimos a la lista.

    if len(new_chunks):                                                 # Si la lista de nuevos chunks no está vacía se dice y se añaden.
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks] # Sacamos los IDs de los nuevos chunks y creamos una lista.
        vector_store.add_documents(new_chunks, ids=new_chunks_ids)      # Es necesario añadir los IDs. Lo pide Chroma.
    else:
        print("No new documents to add.")


# Función para crear los IDs de forma: "Source : Page : Chunk"
# CREACIÓN DE LOS STRING_ID --> chunk_id
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    '''
    Cuando se pone for chunk in chunks_with_ids:, lo que estamos haciendo es definir un bucle for que dura 
    'chunk_with_ids' iteraciones, y con cada iteración se va a tomar un elemento de chunks_with_ids y se asigna 
    a la variable chunk de forma que vamos a poder acceder a sus propiedades y trabajar con él.
    '''
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # Cómo cada página tiene varios chunks creamos el 'id autoincremental'
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id # Compruebas si has cambiado de página y/o documento

        chunk.metadata["id"] = chunk_id

    return chunks


if __name__ == "__main__":
    '''
    import google.generativeai as genai

    genai.configure(api_key=API_KEY_GOOGLE)  # replace with your api key.

    models = [m for m in genai.list_models()]

    for model in models:
        print(model)

    '''

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    '''
    vector_store = Chroma(
        collection_name="instrucciones_google",  # Nombre de la colección
        embedding_function=get_embedding_function.get_embedding_function_google(API_KEY_GOOGLE),  # Función de embedding
        persist_directory=CHROMA_PATH,  # Directorio en el que queremos que se guarde

    )


    # Podemos obtener los embeddings mediante:
    embedding_test = vector_store.get(
        ids=["data/Instrucciones-Bang.pdf:1:1"],
        include=["embeddings"]
    )
    print(len(embedding_test))
    print(embedding_test)
'''
