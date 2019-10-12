import pygame
from random import randint

pygame.init()

x = 0
y = 330
velocidade = 10
velocidade_objetos = 10
gravidade = 10
pos_x = 1010
pos_y = 380

fundo = pygame.image.load('fundo.png')
boneco = pygame.image.load('boneco.png')
gato = pygame.image.load('gato.png')
dog = pygame.image.load('dog.png')
gir = pygame.image.load('gir.png')
rato = pygame.image.load('rato.png')

janela = pygame.display.set_mode((1000,600))
pygame.display.set_caption("Darwin IA006 - Algoritimo Evolutivo")

janela_aberta = True

while janela_aberta:

	pygame.time.delay(40)

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			janela_aberta = False

	comandos = pygame.key.get_pressed()

	if comandos[pygame.K_UP]:
		y -= velocidade

	if comandos[pygame.K_DOWN]:
		y += velocidade

	if comandos[pygame.K_RIGHT]:
		x += velocidade

	if comandos[pygame.K_LEFT]:
		x -= velocidade	

	if (pos_x <= -1000):
		pos_x = 999

	pos_x -= velocidade_objetos

	janela.blit(fundo, (0,0))
	janela.blit(boneco, (x,y))
	janela.blit(gato, (pos_x+ 200, pos_y))
	janela.blit(dog, (pos_x+ 300, pos_y))
	janela.blit(gir, (pos_x+ 400, pos_y))
	janela.blit(rato, (pos_x+ 500, pos_y))


	#pygame.draw.circle(janela, (0,255,0), (x,y), 50)
	pygame.display.update()


pygame.quit()
