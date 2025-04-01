import sqlite3
import numpy as np

from langchain_community.vectorstores.vdms import embedding2bytes

DATABASE_PATH = "chroma/chroma.sqlite3"

'''
def extract_tables():
    # Abrir un archivo de texto para guardar los campos
    with open('campos_de_todas_las_tablas.txt', 'w') as file:
        # Iterar sobre cada tabla
        for table in tables:
            table_name = table[0]  # Nombre de la tabla
            file.write(f"Campos de la tabla: {table_name}\n")

            # Obtener los campos de la tabla actual
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()

            # Escribir los campos en el archivo
            for column in columns:
                column_name = column[1]  # Nombre de la columna
                column_type = column[2]  # Tipo de la columna
                file.write(f"  - {column_name} ({column_type})\n")

            file.write("\n")  # Salto de línea entre tablas

    print("Los campos de todas las tablas han sido guardados en 'campos_de_todas_las_tablas.txt'.")
'''

if __name__ == "__main__":
    # Conectar a la base de datos SQLite
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    '''
    # Obtener la lista de todas las tablas en la base de datos
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    cursor.execute("SELECT block FROM embedding_fulltext_search_data limit 1;")
    embedding_blob = cursor.fetchall()
    print(type(embedding_blob))
    print(embedding_blob)
    print(type(embedding_blob[0]))
    print(embedding_blob[0])
    a = embedding_blob[0]
    '''

    cursor.execute("SELECT vector FROM embeddings_queue where id= 'data/Instrucciones-Bang.pdf:1:1' ")
    result = cursor.fetchall() # fetchall sirve para recuperar todos los datos de una consulta SQL

    for row in result:
        blob_data = row[0]
        vector_bytes = bytes(blob_data)
        vector_array = np.frombuffer(vector_bytes, dtype=np.float32)
        print(vector_array[0:5])
        print((len(vector_array)))


    # Cerrar la conexión
    conn.close()