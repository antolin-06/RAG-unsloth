from click import prompt

from query_data import  query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.2")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower() #eliminamos espacion y hacemos todas las letras minúsculas
    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

def test_ciudadelas_rules():
    assert query_and_validate(
        question="¿Con cuántos distritos termina una partida de ciudadelas? (Contesta solo con un número)",
        expected_response="7",
    )

def test_bang_rules():
    assert query_and_validate(
        question="¿Cuánto puntos de daño recibe un jugador si le explota la carta dinamita en bang? (Contesta solo con un número)",
        expected_response="3",
    )

def test_munchkin_rules():
    assert query_and_validate(
        question="¿Cúantos niveles obtienes tras matar a un monstruo en munchkin? (Contesta solo con un número)",
        expected_response="1",
    )

def test_risk_rules():
    assert query_and_validate(
        question="Si estan jugando 3 jugadores al risk ¿con cuántos peones de infantería empieza cada jugador? (Contesta solo con un número)",
        expected_response="35",
    )