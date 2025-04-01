# RAG. Revisión completa.
###### Adrián Antolín Pellicer
## 1. ¿Qué es la RAG?
La RAG (Retrieval-Augmented-Generation o Generación Mejorada por Recuperación) es el proceso de optimización de salida de un LLM, de forma que haga referencia a una BBDD definida por el propietario, dejando de lado el entrenamiento del LLM. 
[AWS](https://aws.amazon.com/es/what-is/retrieval-augmented-generation/).  
El funcionamiento de un LLM con la RAG, de forma simple, se puede resumir en:
1. El usuario pasa un prompt al LLM.
2. La RAG coge ese prompt y busca información que considera relevante o que coincide de alguna forma con 
dicho prompt.
3. La RAG crea un nuevo prompt a partir de la información recopilada y pide al LLM que responda a la 
petición del usuario con base en esa información.  
[![Diagrama RAG simple](/img_md/rag-fmw.png)](https://www.promptingguide.ai/research/rag)  
Por tanto, si el RAG envía información sin relación o irrelevante, el LLM no debería de responder nada, ya que lo que buscamos es que se conteste únicamente basándose en la información proporcionada y sin usar los conocimientos propios del LLM.

## 2. Leyendo los documentos. Chunking.
Los LLMs no leen el texto como los humanos, por lo que hay que hallar una forma para poder transformar el lenguaje natural en números.  
Podemos destacar 2 procesos iguales, pero diferentes que se emplean para diferentes técnicas:
- Tokenización. En este proceso se divide un texto en tokens que abarcan una palabra, subpalabra o caracteres individuales. Facilitan el procesado y son especialmente útiles para el uso de los mecanismos de atención de los LLMs modernos. La atención ayuda a los LLMs a entender el contexto de una palabra dentro de una frase.
- Chunking. Este proceso es el que usaremos para la RAG. Divide un texto en pedazos más pequeños, como párrados u oraciones. De esta forma al recuperar un chunk, el LLM puede devolver una respuesta bastante más relevante que si solo se hubiera devuelto un token.  

Para empezar con el proceso de chunking, parimos de una serie de documentos (PDFs) que vamos a trocear. Los chunks van a crearse según las especificaciones que definimos en el siguiente fragmento de código dentro de *populate_database.py*.
```python
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
```

### 2.1 ¿Chunks grandes o pequeños?
Lo primero que se me ha ocurrido es que cuánto más grande sean los chunks, menos contexto se perderá y mejor será la respuesta final. Pero en este caso, ¿por qué no pasar siempre el PDF completo? Esto sería contraproducente porque me imagino que llevaría mucho tiempo y mucho procesamiento, lo que quitaría utilidad a la RAG.  
Los chunks deben de cumplir un compromiso entre longitud y calidad de la respuesta, y esto variará dependiendo de la función que queramos darle a nuestra RAG.
- ¿Qué pasa si el chunk es grande? (+1000 palabras)  
En cada búsqueda proporcionaremos más contexto y con menos consultas podremos tener la respuesta completa, pero es más probable que le demos información irrelevante, que sea menos preciso al buscar y también requerirá más memoria.
- ¿Qué pasa si el chunk es pequeño? (-400 palabras)  
Tendremos una mejor precisión, encontraremos respuestas más específicas y será más eficiente. Sin embargo, podremos perder contexto y puede que sean necesarias más consultas para obtener una respuesta completa.

Podemos decir que nos interesa usar chunks grandes cuando el contenido sigue una estructura clara y se debe mantener el contexto. Por ejemplo en un artículo, nos aseguraríamos de que no cortemos la explicación a medias.  
En cambio, los chunks pequeños serían más útiles para una mejor precisión en documentos densos pero con información independiente entre apartados. Un ejemplo sería buscar en una base de datos de FAQ.  

En nuestro caso, como vamos a tratar de resolver dudas de juegos de mesa, creo que lo mejor sería definir chunks pequeños porque las manuales están bien estructurados y así evitamos contexto irrelevante.  
También tenemos que tener en cuenta que al generar los chunks es recomendable generarlos con una pequeña solapación (overlap) para que no se pierda el contexto entre chunks completamente.


## 3. Embeddings.
Los embeddings son la forma de transformar nuestros chunks o tokens en vectores numéricos. Hay muchísimas funciones diferentes y dependiendo de si es más o menos compleja, obtendremos mejores o peores resultados con mayor o menor tiempo de procesamiento.  
Dentro de *get_embedding_function.py*, podemos ver las llamadas a diferentes funciones de embeddings. Lo mejor para buscar es entrar en [Hugging Face](https://huggingface.co/models?sort=trending&search=embedding). Los modelos siempre están etiquetados y he visto que los más comunes para nuestra aplicación son:
- Sentence Similarity: Optimizados para comparar frases y medir similitud semántica. Útil para QA y detección de duplicados, pero no ideal para RAG.
- Feature Extraction: Generan embeddings que pueden usarse para diversas tareas. Se enfocan en representar el significado del texto sin estar limitados a comparar frases.
- Retrieval: Diseñados específicamente para búsqueda semántica en grandes bases de datos. Ideales para RAG, ya que recuperan documentos relevantes basándose en significado, no palabras clave.

Hay que tener en cuenta que necesitamos usar la misma función de embeddings para realizar la búsqueda en la VS posteriormente (lógico).


## 4. Vector Store (VS)
Las Vector Store se pueden definir como bases de datos no relacionales optimizadas para manejar y organizar vectores numéricos de alta dimensión. En nuestro caso vamos a usar Chroma y podemos decir que es del tipo clave-vector, es decir, almacena los vectores de información junto a una clave o ID.  

Para poder entender mejor la VS, vamos a crear nuestros propios IDs y luego añadirlos manualmente junto a los vectores. Para ello hacemos:
```python
# Función para crear los IDs de forma: "Source : Page : Chunk"
# CREACIÓN DE LOS STRING_ID --> chunk_id
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

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
```

### 4.1 ChromaDB.
Ahora, solo faltaría meter la información en la propia VS. Pero vamos a intentar entender todo lo que sucede y los métodos que utilizamos.  
En Chroma tenemos los siguientes conceptos:
- Cliente: permite conectar con chroma, actúa como un intermediario con el que podremos llamar a los diferentes métodos. A su vez, cuando se genera el cliente, se genera la VS, y si no definimos el cliente como persistente, la propia VS no se guardará en el tiempo, es decir, no perdemos los datos al cerrar la aplicación o reiniciar el sistema. También va a crear un directorio de nombre 'CHROMA_PATH' en el que encontraremos la VS, por defecto 'chroma.sqlite3'. Si accedemos a ella vemos que se han creado un montón de tablas diferentes. ¿¿¿Y los otros directorios que se crean automáticamente??? 
- Colecciones: se podría decir que son nuestras BBDD. En nuestro generamos una de instrucciones y almacenará los PDFs asociados. Si quisiéramos guardar información de, por ejemplo, alineaciones de fútbol, podríamos crear otra colección para eso y luego tendríamos que asegurarnos de que se busca en la colección adecuada. En resumen: creamos datasets. Se pueden ver en la tabla de colecciones.

Para añadir información a las colecciones, podemos usar el método de chroma 'collection.add()' o si usamos langchain collection.add_documents(data, ids).




TABLAS DE CHROMA:
- Collections: podemos ver las colecciones que hemos creado.
- Embedding_fulltext_search: vemos los textos de los documentos que hemos guardado (en nuestro caso sería ver los textos de los chunks y se ven todo seguido)
- Embedding_fulltext_search_content: vemos cada documento/chunk con su id. Este id indica el número que ocupa el documento en la tabla, el primero 1, el segundo 2, etc.
- Embeddings_queue: vemos algo extraño porque hay cierta información guardada en binario, que son los embeddings, pero no podemos visualizarlo usando la VS. Esto se debe a que se guardan los embeddings en variables de tipo BLOB (Binary Large Object).



## 5. Testing RAG
### Modelo BAAI/bge-m3 de HuggingFace
#### Preguntas ciudadelas
Utilizando el método de similarity_search_with_score (no entiendo bien lo que hace) y seleccionando los top 5 chunks:  
- ¿Cuál es la forma de ganar en ciudadelas?  
Según la información proporcionada, el jugador con más puntos después de completar 7 distritos en su ciudad se convertirá en el próximo Maestro Constructor del Reino.  
[Parcialmente correcta]
- ¿Cómo se consiguen puntos en ciudadelas?  
Según la información proporcionada, los puntos en Ciudadelas se obtienen al final del juego, cuando cada jugador suma sus distritos construidos. El valor de cada distrito es igual a su costo. Por lo tanto, el jugador con más puntos, que tiene la ciudad medieval más impresionante, se convierte en el próximo Maestro Constructor del Reino.
[Parcialmente correcta]
- Una vez terminada una partida de ciudadelas ¿cómo se calculan los puntos que tiene cada jugador?  
Según el texto proporcionado, los puntos se calculan de la siguiente manera:  
Se suman los puntos equivalentes al costo de construcción de cada uno de los distritos de cada ciudad.  
Si una ciudad tiene por lo menos un distrito de cada tipo (militar, naval y feudal), se suman 3 puntos adicionales.  
Así que el primer paso es calcular la puntuación de los distritos y luego sumar 3 puntos adicionales si las condiciones anteriores se cumplen. [Parcialmente correcta]

#### Preguntas Bang
Utilizando el método de similarity_search_with_score (no entiendo bien lo que hace) le hacemos la siguiente pregunta:    
¿Cuánto daño hace la carta dinamita de bang?  
He variado la cantidad de chunks relevantes para ver como varía la respuesta. (La respuesta que esperamos es que la dinamita hace 3 puntos de daño):  
k=5 -- La información sobre el daño causado por la Dinamita no está explícitamente proporcionada en el texto. Sin embargo, se menciona que "los daños causados por la Dinamita no están provocados por ningún jugador", lo que sugiere que la carta Dinamita tiene un efecto independiente y directo sobre los jugadores que están en el área de su efecto. El texto también menciona que el daño causado por una carta BANG! se cuenta como puntos de vida perdidos, pero no proporciona una cantidad específica para la Dinamita. Por lo tanto, no podemos determinar con certeza cuánto daño hace la carta Dinamita de Bang.  
k=7 -- La respuesta no está explícitamente mencionada en el contexto proporcionado, pero se puede inferir que la Dinamita causa 3 puntos de daño. En efecto, se menciona que "Si esto no sucede, la Dinamita pasa al jugador de su izquierda quien deberá “¡desenfundar!” en su turno. Los jugadores continúan pasando la Dinamita" y "El jugador pierde inmediatamente 3 puntos de vida".  
k=10 -- La carta Dinamita de Bang! hace 3 puntos de daño. Sin embargo, si el jugador que pone en juego la Dinamita roba una carta que muestra Picas y un número entre 2 y 9, ¡la Dinamita explota y el jugador pierde inmediatamente 3 puntos de vida.  

#### Otras preguntas
Pruebo a hacerle preguntas de otros juegos de mesa y otros campos para ver si tiene alucinaciones/se inventa la respuesta.  
¿Cómo puedo hacer un hotel en monopoly?  -- no tiene información para responder
¿Cúantos niveles necesito para ganar en munchkin? -- no tiene información para responder
¿Cúal es la utilidad del oro en munchkin? -- No se menciona explícitamente la utilidad del oro en Munchkin, pero según la información proporcionada, cada jugador comienza con una reserva de oro y utiliza esta reserva para pagar los distritos de construcción y otros efectos. Esto sugiere que el oro tiene un valor como medio de intercambio para acceder a ciertos recursos o beneficios en el juego. Sin embargo, no se proporciona información específica sobre la utilidad del oro en Munchkin.
En general, el oro en juegos como Munchkin se utiliza comúnmente para pagar por cartas o distritos en los que se puede avanzar, mejorar habilidades o acceder a nuevos recursos. Pero sin más contexto, es difícil determinar con precisión su utilidad en este juego específico.  


### Conclusiones de las ejecuciones
Vemos que si la pregunta es un poco más 'abierta' o contiene algo que pueda coincidir con la información de la VS, se puede confundir y si se puede llega a inventar la respuesta. (alucina)  
Observamos que claramente al añadir chunks la respuesta se vuelve más precisa, pero también tarda bastante más en responder. Otra observación del apartado del bang es que como hemos escrito exactamente la misma pregunta los 5 primeros chunks son los mismos siempre, los 7 primeros son iguales, etc.
Pero me quedan otras muchas dudas...
- ¿Con un modelo de embeddings diferente, podríamos obtener unos mejores resultados?
- ¿Es mejor tener chunks más largos o más cortos? [medio contestada en el apartado de los chunks]
- ¿Cúanto nos influye el LLM que procesa el prompt en nuestra respuesta final? ¿Diferentes LLMs, diferentes alucinaciones?
- ¿Cómo de importante es el método de búsqueda? ¿Cúales son los mejores en cuanto a velocidad y precisión?


## 6. Análisis de las variables.
### CHUNKING
Manteniendo el modelo de embeddings BAAI/bge-m3 de HuggingFace.

#### size=300


# ANOTACIONES
link para el apartado de profundización en el rag https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c  
EL METODO DE COMPARACION SE ELIGE EN query_data.py - línea 28. cuando se construye el nuevo prompt  
No he encontrado donde ni como se crea la BBDD. debe de ser una funcion interna de Chroma
TABLAS DE LA BBDD  
siguen el siguiente formato:  
índice | nombre | tipo de dato | notnull | default value | parte de la clave primaria  
collection_metadata - vacio  
collections - número de colecciones que hay  
embeddings - chunks en los que se han dividido los pdfs   
embeddings_queue - texto extraido de cada chunk con más metadatos (id, fecha,) el campo 'vector' (tipo BLOB) saca algo raro que creo que está codificado  
databases                          
embeddings_queue_config          
embedding_fulltext_search          
maintenance_log                  
embedding_fulltext_search_config   
max_seq_id                       
embedding_fulltext_search_content  
migrations                       
embedding_fulltext_search_data     
segment_metadata                 
embedding_fulltext_search_docsize  
segments                         
embedding_fulltext_search_idx      
tenants                          
embedding_metadata           


# RAG-unsloth
