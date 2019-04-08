#!/usr/bin/python
#coding: utf-8
import random

modelo = input("Digite o modelo: ")
tamanho_cromosomo = len(modelo)
tamanho_populacao = 100
geracoes = 50000

def peso_escolhido(items):
  peso_total = sum((item[1] for item in items))
  elemento = random.uniform(0, peso_total)
  for item, peso in items:
    if elemento < peso:
      return item
    elemento = elemento - peso
  return item

#gera caracteres aleatórios para compor a populacao
def caracter_aleatorio():
  return chr(int(random.randrange(32, 255, 1)))

#gera populacoes aleatórias de cromossomos com os caracteres aleatórios anteriores
def populacao_aleatoria():
  populacao = []
  for i in range(tamanho_populacao):
    cromosomo = ""
    for j in range(tamanho_cromosomo):
      cromosomo += caracter_aleatorio()
    populacao.append(cromosomo)
  return populacao

#verifica a forca de um cromossomo, para saber se ele esta próximo ou não do que é esperado no modelo
def fitness(cromosomo):
  fitness = 0
  for i in range(tamanho_cromosomo):
    fitness += abs(ord(cromosomo[i]) - ord(modelo[i]))
  return fitness

#se o cromossomo ja for igual ao do modelo ele é mantido, se for diferente seu valor muda para outro que ainda não tenho passado pelo teste
def mutacao(cromosomo):
  cromossomo_saida = ""
  chance_mutacao = 100
  for i in range(tamanho_cromosomo):
    if int(random.random() * chance_mutacao) == 1:
      cromossomo_saida += caracter_aleatorio()
    else:
      cromossomo_saida += cromosomo[i]
  return cromossomo_saida

#cruzamento ou combinação de dois cromossomos gerando dois novos cromossomos
def crossover(cromosomo1, cromosomo2):
  posicao = int(random.random() * tamanho_cromosomo)
  return (cromosomo1[:posicao] + cromosomo2[posicao:], cromosomo2[:posicao] + cromosomo1[posicao:])

if __name__ == "__main__":

  populacao = populacao_aleatoria()

  for geracao in range(geracoes):
    print("Geração %s | População: '%s'" % (geracao, populacao[0]))
    peso_populacao = []
    if(populacao[0] == modelo):
      break

    for individuo in populacao:
      fitness_valor = fitness(individuo)
      if fitness_valor == 0:
        pares = (individuo, 1.0)
      else:
        pares = (individuo, 1.0 / fitness_valor)
      peso_populacao.append(pares)
    populacao = []

    for i in range(int(tamanho_populacao)):
      individuo1 = peso_escolhido(peso_populacao)
      individuo2 = peso_escolhido(peso_populacao)
      individuo1, individuo2 = crossover(individuo1, individuo2)
      populacao.append(mutacao(individuo1))
      populacao.append(mutacao(individuo2))
  fit_string = populacao[0]
  minimo_fitness = fitness(populacao[0])

  for individuo in populacao:
    fit_individuo = fitness(individuo)

    if fit_individuo <= minimo_fitness:
      fit_string = individuo
      minimo_fitness = fit_individuo

  print("População Final: %s" % fit_string)
