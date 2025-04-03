import pygame
import numpy as np
from random import randint, uniform

pygame.init()

clock = pygame.time.Clock()

# other vars
vel_obj = 10
gravity = 10
pos_x = 1010
pos_y = 380

# images
bg = pygame.image.load('fundo.png')
char_left = pygame.image.load('bonecol.png')
char_right = pygame.image.load('boneco.png')
char = pygame.image.load('boneco.png')

win = pygame.display.set_mode((900, 800))
pygame.display.set_caption("Darwin Algoritimo Evolutivo")

font = pygame.font.SysFont('comicsans', 30, True)
font2 = pygame.font.SysFont('Arial', 21)

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.genome = self.generate_random_genome()
        self.build_weights_from_genome(self.genome)

    def generate_random_genome(self):
        total_weights = (self.input_size * self.hidden_size) + (self.hidden_size * self.output_size)
        return [uniform(-1, 1) for _ in range(total_weights)]

    def build_weights_from_genome(self, genome):
        split1 = self.input_size * self.hidden_size
        w1 = np.array(genome[:split1]).reshape((self.input_size, self.hidden_size))
        w2 = np.array(genome[split1:]).reshape((self.hidden_size, self.output_size))
        self.w1 = w1
        self.w2 = w2

    def forward(self, inputs):
        h = np.dot(inputs, self.w1)
        h = np.tanh(h)
        out = np.dot(h, self.w2)
        return out

# Dino agent with brain
class DinoAgent:
    def __init__(self, x, y, width, height):
        self.player = player(x, y, width, height)
        self.brain = NeuralNetwork(3, 6, 1)  # 3 inputs → 6 hidden → 1 output
        self.fitness = 0

    def update(self, obstacle):
        # Inputs: distance to obstacle, height of obstacle, speed
        dist = obstacle.x - self.player.x
        height = obstacle.y  # or fixed if all are same
        speed = obstacle.vel

        inputs = np.array([dist / 900, height / 800, speed / 10])  # normalize
        decision = self.brain.forward(inputs)

        if decision[0] > 0 and not self.player.is_jump:
            self.player.is_jump = True

        if self.player.is_jump:
            if self.player.jump_count >= -10:
                neg = 1
                if self.player.jump_count < 0:
                    neg = -1
                self.player.y -= (self.player.jump_count ** 2) * 0.8 * neg
                self.player.jump_count -= 1
            else:
                self.player.is_jump = False
                self.player.jump_count = 10

        self.fitness += 1  # basic survival fitness

    def draw(self, win):
        self.player.draw(win)

# Original player and enemy classes
class player:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 5
        self.is_jump = False
        self.jump_count = 10
        self.left = False
        self.right = False
        self.walk_count = 0
        self.box = (self.x, self.y, 100, 150)

    def draw(self, win):
        if self.walk_count + 1 >= 27:
            self.walk_count = 0

        if self.left:
            win.blit(char_left, (self.x, self.y))
            self.walk_count += 1
        elif self.right:
            win.blit(char_right, (self.x, self.y))
            self.walk_count += 1
        else:
            win.blit(char, (self.x, self.y))

        self.box = (self.x + 10, self.y, 100, 150)
        pygame.draw.rect(win, (255, 0, 0), self.box, 2)

class enemy:
    walk_left = [pygame.image.load('gato.png'), pygame.image.load('dog.png'), pygame.image.load('gir.png'), pygame.image.load('rato.png')]

    def __init__(self, x, y, width, height, end):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.path = [self.x, self.y]
        self.end = end
        self.walk_count = 0
        self.vel = 7
        self.box = (self.x, self.y, 10, 50)

    def draw(self, win):
        self.move()
        if self.walk_count + 1 <= 33:
            self.walk_count = 0
        win.blit(self.walk_left[self.walk_count // 3], (self.x, self.y))
        self.walk_count += 1
        self.box = (self.x, self.y, 65, 84)
        pygame.draw.rect(win, (255, 0, 0), self.box, 2)

    def move(self):
        if self.x > -10:
            self.x -= self.vel
        else:
            self.x = 900

    def stop(self):
        self.x = 500
        self.vel = 0

def redraw_game_window(dino, obstacle):
    win.blit(bg, (0, 0))
    text = font.render('Pontuacao: ' + str(dino.fitness), 1, (0, 0, 0))
    win.blit(text, (10, 30))
    dino.draw(win)
    obstacle.draw(win)
    pygame.display.update()

# Start with one agent for now
dino = DinoAgent(10, 350, 64, 64)
gobling = enemy(900, 400, 64, 64, 300)
run = True

while run:
    clock.tick(40)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    dino.update(gobling)
    redraw_game_window(dino, gobling)

pygame.quit()

