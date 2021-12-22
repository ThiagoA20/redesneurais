import pandas as pd
import os
import time
import random

def Soma(lista_entradas, pesos):
    # Σ = (x1 * w1) + (x2 * w2) + ... + (xn * wn)
    soma_resultado = 0

    for i in range(len(lista_entradas)):
        soma_resultado += lista_entradas[i] * pesos[i]

    return soma_resultado

def Ativacao(resultado_soma):
    # Step function: f(x) = 1 if x >= 1 else 0
    if resultado_soma >= 1:
        return 1
    else:
        return 0

def CalcularErro(resultado_esperado, resultado_obtido):
    # Erro = Saída Esperada - Resultado Obtido
    erro = resultado_esperado - resultado_obtido
    return erro

taxa_de_aprendizado = 0.1

def Ajuste(peso, erro, entrada):
    # Ajuste = peso + (taxa de aprendizado * erro * entrada)
    ajuste_resultado = peso + (taxa_de_aprendizado * erro * entrada)

    return ajuste_resultado

tempos_de_execucao = []
for u in range(100):
    start_time = time.time()
    df = pd.read_csv("and_table.csv", sep=";")

    peso = random.random()
    peso2 = random.random()
    eficiencia = 0
    pesos = [peso, peso2]
    geracao = 0
    while eficiencia < 90:

        os.system("cls")
        eficiencia = 100

        for i in range(len(df.values)):
            lista_entradas = [df.values[i][0], df.values[i][1]]
            resultado_soma = Soma(lista_entradas, pesos)
            resultado_obtido = Ativacao(resultado_soma)
            erro = CalcularErro(df.values[i][2], resultado_obtido)

            if erro != 0:
                eficiencia -= 25
                for i in range(len(lista_entradas)):
                    ajuste = Ajuste(pesos[i], erro, lista_entradas[i])
                    pesos[i] = ajuste
            
            print(f"Entradas: ({df.values[i][0]},{df.values[i][1]}) / Soma: {resultado_soma} / Ativação: {resultado_obtido} / Erro: {erro} / Peso: {pesos}")
        
        geracao += 1
    
    print(f"Geração: {geracao}")

    tempo_de_execucao = time.time() - start_time

    tempos_de_execucao.append(tempo_de_execucao)

media = 0
for i in range(len(tempos_de_execucao)):
    media += tempos_de_execucao[i]

media = media / len(tempos_de_execucao)

print(f"Media de tempo de execução: {media}")