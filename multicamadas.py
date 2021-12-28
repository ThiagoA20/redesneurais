# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from itertools import count
# import numpy as np
# import random

# plt.style.use("fivethirtyeight")

# x_vals = []
# y_vals = []

# index = count()

# plt.xlim(xmin=3)
# plt.ylim(ymin=3)


# def Animate(i):
#     x_vals.append(next(index))
#     y_vals.append(random.randint(0, 10))

#     plt.cla()
#     plt.plot(x_vals, y_vals)


# ani = FuncAnimation(plt.gcf(), Animate, interval=1000)

# plt.tight_layout()
# plt.show()


# Deep learning tópicos: gradiente desaparecendo, outras funções de ativação, redes neurais convolucionais, redes neurais recorrentes, Keras, Theano, TensorFlow, programação GPU
# Quantos neurônios numa camada oculta? neuronios = entradas + saidas / 2
# Validação cruzada, AutoML, em geral duas camadas funcionam bem para poucos dados
# conversão de dados - estatística
# Recomendação: RELU camada oculta, sigmoide camada de saida

import math
import pandas as pd
import numpy as np
import random
import os

df = pd.read_csv("xor.csv", sep=';')

class NeuralNetwork:


    def __init__(self, n_ocultas,  n_entradas, n_saidas, ativacao_oculta="ReLU", ativacao_saida="Sigmoide", pesos_file="pesos.csv"):
        self.n_camadas_ocultas = n_ocultas
        self.n_neuronios_saida = n_saidas
        self.n_neuronios_entrada = n_entradas
        self.ativacao_oculta = ativacao_oculta
        self.ativacao_saida = ativacao_saida
        self.pesos_file = pesos_file

        self.n_neuronios_oculta = round((self.n_neuronios_entrada + self.n_neuronios_saida) / 2)
        self.pesos_oculta = []
        self.pesos_saida = []


    def configuracaoNeuronios(self):
        configuracao = f"Entradas: {self.n_neuronios_entrada} / Número de camadas ocultas: {self.n_camadas_ocultas} / Neurônios na camada oculta: {self.n_neuronios_oculta} / Saídas: {self.n_neuronios_saida}"
        return configuracao


    def SalvarPesos(self, **kwargs):
        """Função para criar um arquivo csv com os valores dos pesos caso não exista, caso exista, sobreescreve os valores. É necessario prover o nome do arquivo, se não ele presume que é pesos.csv"""
        try:
            filename = kwargs["filename"]
        except KeyError:
            filename = "pesos.csv"

        with open(os.path.join("neural_network_knowledge", filename), 'w') as database:
            database.write("camada;neuronio;posicao;valor_do_peso\n")

            for camada in range(len(self.pesos_oculta)):
                for neuronio in range(len(self.pesos_oculta[camada])):
                    for peso in range(len(self.pesos_oculta[camada][neuronio])):
                        database.write(f"{camada};{neuronio};{peso};{self.pesos_oculta[camada][neuronio][peso]}\n")

            for neuronio in range(len(self.pesos_saida)):
                for peso in range(len(self.pesos_saida[neuronio])):
                    database.write(f"{len(self.pesos_oculta)};{neuronio};{peso};{self.pesos_saida[neuronio][peso]}\n")

            database.close()


    def LerPesos(self, **kwargs):
        """Função que lê os pesos do arquivo csv e retorna os pesos de cada camada, é necessario passar como argumento o nome do arquivo, caso contrário ele vai presumir que é pesos.csv"""
        pesos_saida = []
        pesos_oculta = []

        try:
            filename = kwargs["filename"]
        except KeyError:
            filename = "pesos.csv"
        
        try:
            database = pd.read_csv(os.path.join("neural_network_knowledge", filename), sep=";")

            pesos = []

            for line in database.values:
                if line[0] == len(pesos):
                    pesos.append([[[]]])
                if line[1] == len(pesos[line[0]]):
                    pesos[line[0]].append([])
                if line[2] == len(pesos[line[0]][line[1]]):
                    pesos[line[0]][line[1]].append(0)

                pesos[line[0]][line[1]][line[2]] = line[3]
                
            pesos_saida = pesos[-1]
            pesos_oculta = pesos[0:-1]

            return pesos_saida, pesos_oculta
        
        except FileNotFoundError:
            return pesos_saida, pesos_oculta


    def iniciarPesos(self):
        """Cria pesos aleatórios, caso não existam e salva na base de dados especificada. Caso existam, lê da base de dados especificada e salva nas variáveis pesos_oculta e pesos_saida"""
        try:
            with open(os.path.join("neural_network_knowledge", self.pesos_file)):
                self.pesos_saida, self.pesos_oculta = self.LerPesos(filename=self.pesos_file)
        except FileNotFoundError:
            self.pesos_oculta = [[[random.randint(-1000, 1000) for u in range(self.n_neuronios_entrada)] for i in range(self.n_neuronios_oculta)] for i in range(self.n_camadas_ocultas)]
            self.pesos_saida = [[random.randint(-1000, 1000) for u in range(len(self.pesos_oculta[-1]))] for i in range(self.n_neuronios_saida)]

            self.SalvarPesos(filename=self.pesos_file)
    

    def Soma(self, entradas, pesos):
        """Multiplica todas as entradas pelos respectivos pesos e retorna a soma de tudo"""
        soma = 0
        for i in range(len(entradas)):
            soma += entradas[i] * pesos[i]
        
        return soma


    def Ativacao(self, resultado_soma, camada):
        """Aplica a função de ativação no resultado obtido da soma"""

        if camada == "oculta":
            ativacao = max(0, resultado_soma)
        elif camada == "saida":
            ativacao = 1 / (1 + np.exp(-resultado_soma))
        else:
            pass

        return ativacao

    
    def CalcularErro(self, ativacao_saida, resultado_esperado):
        erro = int(resultado_esperado) - int(ativacao_saida)
        return erro
        
    

    def feedFoward(self, lista_de_entradas, resultados_esperados):
        """Passa todas as entradas pelas funções de soma e ativação e retorna o valor da ativação de cada neurônio e o erro calculado"""
        Erro = 0

        resultado_neuronios = []

        if len(lista_de_entradas) == self.n_neuronios_entrada and len(resultados_esperados) == self.n_neuronios_saida:
            i = 0
            while i < len(self.pesos_oculta):
                resultado_neuronios.append([])
                # print(self.pesos_oculta[i])
                if i == 0:
                    for neuronio in range(len(self.pesos_oculta[i])):
                        soma = self.Soma(lista_de_entradas, self.pesos_oculta[i][neuronio])
                        ativacao = self.Ativacao(soma, "oculta")
                        resultado_neuronios[i].append(ativacao)
                        print(f"Entradas: {lista_de_entradas} / Pesos: {self.pesos_oculta[i][neuronio]} / Soma: {soma} / Ativação: {ativacao}")
                elif i == len(self.pesos_oculta) - 1:
                    for neuronio in range(len(self.pesos_oculta[i])):
                        soma = self.Soma(resultado_neuronios[i - 1], self.pesos_oculta[i][neuronio])
                        ativacao = self.Ativacao(soma, "oculta")
                        resultado_neuronios[i].append(ativacao)
                        print(f"Entradas: {lista_de_entradas} / Pesos: {self.pesos_oculta[i][neuronio]} / Soma: {soma} / Ativação: {ativacao}")
                else:
                    for neuronio in range(len(self.pesos_oculta[i])):
                        soma = self.Soma(resultado_neuronios[i - 1], self.pesos_oculta[i][neuronio])
                        ativacao = self.Ativacao(soma, "oculta")
                        resultado_neuronios[i].append(ativacao)
                        print(f"Entradas: {lista_de_entradas} / Pesos: {self.pesos_oculta[i][neuronio]} / Soma: {soma} / Ativação: {ativacao}")
                i += 1

            resultado_neuronios.append([])
            for neuronio in range(len(self.pesos_saida)):
                soma = self.Soma(resultado_neuronios[0], self.pesos_saida[neuronio])
                ativacao = self.Ativacao(soma, "saida")
                resultado_neuronios[-1].append(ativacao)
                print(f"Entradas: {resultado_neuronios[0]} / Pesos: {self.pesos_saida[neuronio]} / Soma: {soma} / Ativação: {ativacao}")
            
            Erro = self.CalcularErro(resultado_neuronios[-1][0], resultados_esperados[0])
        else:
            Erro = "Valores incompatíveis com a instância atual!"
        
        print(resultado_neuronios)
        
        return Erro


    def backpropagation(self):
        """Calcula o delta da camada de saída, camada oculta e atualiza os pesos"""
        pass



n1 = NeuralNetwork(1, 2, 1, pesos_file="xor_pesos.csv")
n1.iniciarPesos()

# print(n1.configuracaoNeuronios())
erro = n1.feedFoward([1, 0], [1])
print(f"Erro: {erro}")


# receber inputs ------------------------------------------------------------------------------------------------------- [X]
# definir aleatoriamente os pesos -------------------------------------------------------------------------------------- [X]
# multiplicar inputs por pesos ----------------------------------------------------------------------------------------- [X]

# somar todas as multiplicações e o bias ------------------------------------------------------------------------------- [ ]
# O bias é sempre 1 e também tem pesos associados

# Aplicar função de ativação ------------------------------------------------------------------------------------------- [X]
# definir pesos da camada oculta --------------------------------------------------------------------------------------- [X]
# multiplicar resultado da ativação pelos pesos ocultos ---------------------------------------------------------------- [X]
# somar todas as multiplicações ---------------------------------------------------------------------------------------- [X]
# aplicar a função de ativação da camada de saída ---------------------------------------------------------------------- [X]
# Comparar resultado obtido com resultado esperado --------------------------------------------------------------------- [ ]

# Calcular o erro ------------------------------------------------------------------------------------------------------ [ ]
# Erro = Resultado esperado - Resultado obtido
# MSE --> mean square error
# RMSE --> root mean square error

# Calcular o delta da camada de saída ---------------------------------------------------------------------------------- [ ]
# DeltaSaída = Erro - Derivada da função de ativação

# Calcular o delta da camada oculta ------------------------------------------------------------------------------------ [ ]
# DeltaEscondida = DerivadaSigmoide * peso (do lado direito) * DeltaSaída

# Backpropagation ------------------------------------------------------------------------------------------------------ [ ]
# Ajuste = (peso * momento) + (Soma(função de ativação * delta saída p/ cada registro) * taxa de aprendizagem)
# esse calculo de ajuste é feito para cada neurônio

# Criar o gradiente