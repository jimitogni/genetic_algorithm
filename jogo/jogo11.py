import pygame
import numpy as np
from random import randint, uniform, choice
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab
import sys

pygame.init()
clock = pygame.time.Clock()

win = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("Darwin Algoritimo Evolutivo")

font = pygame.font.SysFont('comicsans', 30, True)
font2 = pygame.font.SysFont('Arial', 21)

# Genetic Algorithm parameters
POPULATION_SIZE = 20
MUTATION_RATE = 0.1

# Track fitness history for plotting
best_fitness_history = []
avg_fitness_history = []

# Enemy colors (geometric shapes)
ENEMY_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

class enemy:
    def __init__(self, x, y, width=40, height=60, end=300, speed=9):
        self.active = True
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.path = [self.x, self.y]
        self.end = end
        self.walk_count = 0
        self.base_speed = speed
        self.vel = speed
        self.box = (self.x, self.y, 10, 50)
        self.color = choice(ENEMY_COLORS)

    def draw(self, win):
        self.move()
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.height))
        self.box = (self.x, self.y, self.width, self.height)
        pygame.draw.rect(win, (255, 0, 0), self.box, 2)

    def move(self):
        self.vel = self.base_speed
        self.x -= self.vel
        if self.x + self.width < 0:
            self.active = False

    def stop(self):
        self.x = 500
        self.vel = 0

# Flying enemy class
class FlyingEnemy(enemy):
    def __init__(self, x, width=40, height=40, speed=9):
        y = 480
        end = 300
        super().__init__(x, y=y, width=width, height=height, end=end, speed=speed)
        self.color = (100, 100, 255)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, genome=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if genome:
            self.genome = genome
        else:
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
        return h, out

class DinoAgent:
    def __init__(self, x, y, width, height, genome=None):
        self.player = player(x, y, width, height)
        self.brain = NeuralNetwork(4, 6, 1, genome)
        self.fitness = 0
        self.alive = True
        self.last_inputs = None
        self.last_hidden = None

    def update(self, obstacle):
        if not self.alive:
            return

        dist = obstacle.x - self.player.x
        height = obstacle.y
        speed = obstacle.vel
        obstacle_type = 1 if isinstance(obstacle, FlyingEnemy) else 0

        inputs = np.array([dist / 900, height / 800, speed / 10, obstacle_type])
        h, decision = self.brain.forward(inputs)

        self.last_inputs = inputs
        self.last_hidden = h

        if decision[0] > 0 and not self.player.is_jump:
            self.player.is_jump = True
            self.player.vertical_velocity = self.player.jump_velocity


        if self.player.is_jump:
            self.player.vertical_velocity += self.player.gravity
            self.player.y += self.player.vertical_velocity

            if self.player.y >= 550:
                self.player.y = 550
                self.player.is_jump = False
                self.player.vertical_velocity = 0

        self.fitness += 1

    def check_collision(self, obstacle):
        if self.player.x + 100 > obstacle.x and self.player.x < obstacle.x + obstacle.width:
            if self.player.y + 150 > obstacle.y:
                self.alive = False

    def draw(self, win):
        if self.alive:
            self.player.draw(win)

class player:
    def __init__(self, x, y, width=40, height=60):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = 5
        self.is_jump = False
        self.vertical_velocity = 0
        self.jump_velocity = -20
        self.gravity = 0.9
        self.char_img = pygame.image.load('boneco.png')
        self.box = (self.x, self.y, 100, 150)

    def draw(self, win):
        win.blit(self.char_img, (self.x, self.y))
        self.box = (self.x + 10, self.y, 100, 150)
        pygame.draw.rect(win, (255, 0, 0), self.box, 2)

def draw_background(surface):
    surface.fill((255, 255, 255))
    pygame.draw.rect(surface, (180, 180, 180), (0, 700, 1200, 100))
    pygame.draw.ellipse(surface, (230, 230, 230), (200, 100, 120, 60))
    pygame.draw.ellipse(surface, (230, 230, 230), (210, 80, 140, 80))
    pygame.draw.ellipse(surface, (230, 230, 230), (900, 50, 150, 70))
    pygame.draw.ellipse(surface, (230, 230, 230), (880, 70, 130, 60))

def draw_network(nn, inputs, hidden, win):
    start_x = 1000
    start_y = 100
    spacing_y = 40
    for i, val in enumerate(inputs):
        color = (255, 165, 0) if val > 0 else (100, 100, 100)
        pygame.draw.circle(win, color, (start_x, start_y + i * spacing_y), 10)
    for j, val in enumerate(hidden):
        color = (255, 0, 0) if val > 0 else (0, 0, 0)
        pygame.draw.circle(win, color, (start_x + 60, start_y + j * spacing_y), 10)
    for i in range(nn.input_size):
        for j in range(nn.hidden_size):
            color = (255, 0, 0) if nn.w1[i][j] > 0 else (0, 0, 255)
            pygame.draw.line(win, color, (start_x, start_y + i * spacing_y), (start_x + 60, start_y + j * spacing_y), 1)
    out_color = (0, 255, 0)
    pygame.draw.circle(win, out_color, (start_x + 120, start_y + 2 * spacing_y), 10)
    for j in range(nn.hidden_size):
        color = (255, 0, 0) if nn.w2[j][0] > 0 else (0, 0, 255)
        pygame.draw.line(win, color, (start_x + 60, start_y + j * spacing_y), (start_x + 120, start_y + 2 * spacing_y), 1)

def evolve_population(old_agents):
    sorted_agents = sorted(old_agents, key=lambda a: a.fitness, reverse=True)
    best = sorted_agents[0].fitness
    avg = sum(a.fitness for a in old_agents) / len(old_agents)
    best_fitness_history.append(best)
    avg_fitness_history.append(avg)
    top_genomes = [a.brain.genome for a in sorted_agents[:5]]
    return [DinoAgent(10 + i * 5, 550, 64, 64, mutate(choice(top_genomes))) for i in range(POPULATION_SIZE)]

def mutate(genome):
    return [gene + uniform(-0.5, 0.5) if uniform(0, 1) < MUTATION_RATE else gene for gene in genome]

def redraw_game_window(agents, enemies, generation, fps):
    draw_background(win)
    alive = sum(agent.alive for agent in agents)
    win.blit(font.render(f'Generation: {generation}', 1, (0, 0, 0)), (10, 60))
    win.blit(font.render(f'Alive: {alive}', 1, (0, 0, 0)), (10, 80))
    win.blit(font.render(f'FPS: {int(fps)}', 1, (0, 0, 0)), (10, 100))
    win.blit(font.render(f'Best Score (All Time): {best_score}', 1, (0, 0, 0)), (10, 120))
    win.blit(font.render(f'Best Score (This Epoch): {epoch_best_score}', 1, (0, 0, 0)), (10, 140))
    win.blit(font.render(f'Speed: {speed_level}', 1, (0, 0, 0)), (10, 160))

    if agents:
        win.blit(font.render(f'Jump Velocity: {agents[0].player.jump_velocity}', 1, (0, 0, 0)), (400, 60))
        win.blit(font.render(f'Gravity: {agents[0].player.gravity}', 1, (0, 0, 0)), (400, 80))
        win.blit(font.render(f'Vertical Velocity: {int(agents[0].player.vertical_velocity)}', 1, (0, 0, 0)), (400, 100))

    for e in enemies:
        e.draw(win)
    for agent in agents:
        agent.draw(win)
    alive_agents = [a for a in agents if a.alive]
    if alive_agents:
        best = max(alive_agents, key=lambda a: a.fitness)
        if best.last_inputs is not None and best.last_hidden is not None:
            draw_network(best.brain, best.last_inputs, best.last_hidden, win)
    pygame.display.update()

def try_spawn_enemy(enemies, last_spawn_time, speed_level, current_time):
    # Only spawn a new enemy if enough time has passed
    if current_time - last_spawn_time > randint(60, 150):  # Frames (1–2.5 seconds at 60 FPS)
        last_x = max([e.x for e in enemies], default=0)
        if last_x < 1000:  # Avoid stacking too close
            base_x = randint(1200, 1600)
            if randint(0, 1):
                enemies.append(enemy(base_x, 636, speed=speed_level))
            else:
                enemies.append(FlyingEnemy(base_x, speed=speed_level))
            return current_time  # Update last_spawn_time
    return last_spawn_time

# Game loop
agents = [DinoAgent(10 + i * 5, 550, 64, 64) for i in range(POPULATION_SIZE)]
generation = 1
speed_level = 9
enemies = []
frame_counter = 0
run = True
best_score = 0
speed_level = 9
epoch_best_score = 0
score_timer = 0
last_spawn_time = 0  # For controlling enemy spawn timing

while run:
    fps = clock.get_fps()
    clock.tick(60)
    frame_counter += 1
    score_timer += 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Increase difficulty every 400 frames
    if frame_counter % 400 == 0:
        speed_level = min(speed_level + 1, 100)

    # Try spawning a new enemy
    last_spawn_time = try_spawn_enemy(enemies, last_spawn_time, speed_level, frame_counter)

    for e in enemies:
        e.base_speed = speed_level

    all_dead = True
    for agent in agents:
        if agent.alive:
            agent.fitness += 1  # Increment fitness every frame the agent is alive
            if agent.fitness > epoch_best_score:
                epoch_best_score = agent.fitness
            all_dead = False
        for e in enemies:
            agent.update(e)
            agent.check_collision(e)

    enemies = [e for e in enemies if e.active]
    redraw_game_window(agents, enemies, generation, fps)

    if all_dead:
        best_agent = max(agents, key=lambda a: a.fitness)
        if best_agent.fitness > best_score:
            best_score = best_agent.fitness

        generation += 1
        agents = evolve_population(agents)
        for agent in agents:
            agent.player.y = 550
        speed_level = 9
        enemies = []
        frame_counter = 0

pygame.quit()

