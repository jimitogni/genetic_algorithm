#!/usr/bin/python
#coding: utf-8
import random

print("Modelo de entrada:")
print("a b c d e f g h i ... j")
print("Digite seu modelo:")
modelo_entrada = input()
modelo = [int(i) for i in modelo_entrada.split()]
print("\n")
print("Modelo: {}".format(modelo))
print("\n")
tam_individuo = len(modelo)
tam_populacao = 10
pais = 2
prob_mutacao = 0.5

def individuo(min, max):
	return[random.randint(min, max) for i in range(tam_individuo)]

def criarPopulacao():
	return[individuo(0,9) for i in range(tam_populacao)]

def funcaoFitness(individuo):
	fitness = 0
	for i in range(len(individuo)):
		if(individuo[i] == modelo[i]):
			fitness += 1
	return fitness

def selecaoEReproducao(populacao):
	pontuados = [(funcaoFitness(i), i) for i in populacao]
	pontuados = [i[1] for i in sorted(pontuados)]
	populacao = pontuados

	selecionados = pontuados[(len(pontuados) - pais):]

	for i in range(len(populacao) - pais):
		ponto = random.randint(1, tam_individuo - 1)
		pai = random.sample(selecionados, 2)

		populacao[i][:ponto] = pai[0][:ponto]
		populacao[i][ponto:] = pai[1][ponto:]

	return populacao

def mutacao(populacao):
	for i in range(len(populacao) - pais):
		if(random.random() <= prob_mutacao):
			ponto = random.randint(0, tam_individuo -1)
			novo_valor = random.randint(1,9)

			while(novo_valor == populacao[i][ponto]):
				novo_valor = random.randint(1,9)


			populacao[i][ponto] = novo_valor

	return populacao

populacao = criarPopulacao()
print("População Inicial: {}".format(populacao))
print("\n")
for i in range(100):
	populacao = selecaoEReproducao(populacao)
	populacao = mutacao(populacao)
print("População Final: {}".format(populacao))