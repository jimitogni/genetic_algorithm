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
        self.color = (100, 100, 255)  # bluish to differentiate  # bluish to differentiate
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

# Neural Network class
# ... (unchanged)
# DinoAgent class

# Neural Network class
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

# Dino agent with brain
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

        if self.player.is_jump:
            if self.player.jump_count >= -20:
                neg = 1
                if self.player.jump_count < 0:
                    neg = -1
                self.player.y -= (self.player.jump_count ** 2) * 0.2 * neg
                self.player.jump_count -= 1
            else:
                self.player.is_jump = False
                self.player.jump_count = 20
            # Clamp to ground level
            if self.player.y > 550:
                self.player.y = 550

        self.fitness += 1

    def check_collision(self, obstacle):
        if self.player.x + 100 > obstacle.x and self.player.x < obstacle.x + 65:
            if self.player.y + 150 > obstacle.y:
                self.alive = False

    def draw(self, win):
        if self.alive:
            self.player.draw(win)

# Original player and enemy classes
class player:
    def __init__(self, x, y, width=40, height=60):
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
        self.char_img = pygame.image.load('boneco.png')
        self.box = (self.x, self.y, 100, 150)

    def draw(self, win):
        win.blit(self.char_img, (self.x, self.y))
        self.box = (self.x + 10, self.y, 100, 150)
        pygame.draw.rect(win, (255, 0, 0), self.box, 2)

def draw_network(nn, inputs, hidden, win):
    start_x = 1000
    start_y = 100
    spacing_y = 40

    # Draw input neurons
    for i, val in enumerate(inputs):
        color = (255, 165, 0) if val > 0 else (100, 100, 100)
        pygame.draw.circle(win, color, (start_x, start_y + i * spacing_y), 10)

    # Draw hidden neurons
    for j, val in enumerate(hidden):
        color = (255, 0, 0) if val > 0 else (0, 0, 0)
        pygame.draw.circle(win, color, (start_x + 60, start_y + j * spacing_y), 10)

    # Draw lines between inputs and hidden layer
    for i in range(nn.input_size):
        for j in range(nn.hidden_size):
            color = (255, 0, 0) if nn.w1[i][j] > 0 else (0, 0, 255)
            pygame.draw.line(win, color, (start_x, start_y + i * spacing_y), (start_x + 60, start_y + j * spacing_y), 1)

    # Draw output neuron
    out_color = (0, 255, 0)
    pygame.draw.circle(win, out_color, (start_x + 120, start_y + 2 * spacing_y), 10)
    for j in range(nn.hidden_size):
        color = (255, 0, 0) if nn.w2[j][0] > 0 else (0, 0, 255)
        pygame.draw.line(win, color, (start_x + 60, start_y + j * spacing_y), (start_x + 120, start_y + 2 * spacing_y), 1)


def draw_background(surface):
    # Sky
    surface.fill((255, 255, 255))

    # Floor
    pygame.draw.rect(surface, (180, 180, 180), (0, 700, 1200, 100))

    # Clouds (simple ellipses)
    pygame.draw.ellipse(surface, (230, 230, 230), (200, 100, 120, 60))
    pygame.draw.ellipse(surface, (230, 230, 230), (210, 80, 140, 80))
    pygame.draw.ellipse(surface, (230, 230, 230), (900, 50, 150, 70))
    pygame.draw.ellipse(surface, (230, 230, 230), (880, 70, 130, 60))

fitness_plot_surface = None

def draw_fitness_plot(best_history, avg_history):
    fig = pylab.figure(figsize=[4, 1.8], dpi=100)
    ax = fig.gca()
    ax.plot(best_history, color="blue", label="Best")
    ax.plot(avg_history, color="red", label="Average")
    ax.legend()
    ax.set_title("Fitness por Geração")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    pylab.close(fig)
    return surf

def redraw_game_window(agents, obstacle, generation, fps, best_score, epoch_best_score):
    draw_background(win)

    # Draw basic info
    text = font.render(f'Generation: {generation}', 1, (0, 0, 0))
    win.blit(text, (10, 10))
    alive = sum(agent.alive for agent in agents)
    text2 = font.render(f'Alive: {alive}', 1, (0, 0, 0))
    win.blit(text2, (10, 30))
    text3 = font.render(f'Speed: {speed_level}', 1, (0, 0, 0))
    text4 = font.render(f'Best Score (All Time): {best_score}', 1, (0, 0, 0))
    text5 = font.render(f'Best Score (This Epoch): {epoch_best_score}', 1, (0, 0, 0))
    win.blit(text4, (10, 50))
    win.blit(text5, (10, 70))
    win.blit(text3, (10, 90))

    for agent in agents:
        agent.draw(win)
    for obs in enemies[:]:
        if obs.active:
            obs.draw(win)
        else:
            enemies.remove(obs)

    alive_agents = [a for a in agents if a.alive]
    if alive_agents:
        best = max(alive_agents, key=lambda a: a.fitness)
        if best.last_inputs is not None and best.last_hidden is not None:
            draw_network(best.brain, best.last_inputs, best.last_hidden, win)


    text6 = font.render(f'FPS: {int(fps)}', 1, (0, 0, 0))
    win.blit(text6, (10, 110))
    pygame.display.update()

def evolve_population(old_agents):
    sorted_agents = sorted(old_agents, key=lambda a: a.fitness, reverse=True)
    best = sorted_agents[0].fitness
    avg = sum(a.fitness for a in old_agents) / len(old_agents)
    best_fitness_history.append(best)
    avg_fitness_history.append(avg)

    new_agents = []
    top_genomes = [a.brain.genome for a in sorted_agents[:5]]
    for i in range(POPULATION_SIZE):
        parent = choice(top_genomes)
        child_genome = mutate(parent)
        new_agents.append(DinoAgent(10 + i * 5, 350, 64, 64, child_genome))
    return new_agents

def mutate(genome):
    new_genome = []
    for gene in genome:
        if uniform(0, 1) < MUTATION_RATE:
            new_genome.append(gene + uniform(-0.5, 0.5))
        else:
            new_genome.append(gene)
    return new_genome

def increase_difficulty(generation):
    speed = 9  # Start with base speed
    enemies = []
    x_pos = 1200
    for _ in range(randint(5, 7)):
        gap = randint(1000, 1500)
        x_pos += gap
        enemy1 = enemy(x_pos, 636, speed=speed_level)
        enemies.append(enemy1)
        # 25% chance to add a second enemy very close
        if uniform(0, 1) < 0.25:
            x_pos += randint(40, 80)
            enemy2 = enemy(x_pos, 636, speed=speed_level)
            enemies.append(enemy2)
    for e in enemies:
        e.vel = speed
    return enemies

# Initial population
generation = 1
agents = [DinoAgent(10 + i * 5, 550, 64, 64) for i in range(POPULATION_SIZE)]  # on the ground
enemies = []
enemy_spawn_timer = 0
next_enemy_spawn_frame = randint(90, 225)  # adjusted for 60 FPS
run = True

frame_counter = 0
score_timer = 0
epoch_best_score = 0
best_score = 0
speed_level = 9

while run:
    fps = clock.get_fps()
    enemy_spawn_timer += 1

    # Spawn new enemy during gameplay
    if enemy_spawn_timer >= next_enemy_spawn_frame:
        base_x = 1200
        if uniform(0, 1) < 0.3:
            new_enemy = FlyingEnemy(base_x, width=40, height=40, speed=speed_level)
            is_flying = True
        else:
            new_enemy = enemy(base_x, 636, speed=speed_level, width=40, height=60)
            is_flying = False
        new_enemy.vel = speed_level
        enemies.append(new_enemy)

        # 25% chance to spawn second close enemy only if there's enough space
        if uniform(0, 1) < 0.25:
            spacing = randint(120, 200) if is_flying else randint(40, 80)
            if uniform(0, 1) < 0.3:
                second_enemy = FlyingEnemy(base_x + spacing, width=40, height=40, speed=speed_level)
            else:
                second_enemy = enemy(base_x + spacing, 636, speed=speed_level, width=40, height=60)
            second_enemy.vel = speed_level
            enemies.append(second_enemy)

        enemy_spawn_timer = 0
        next_enemy_spawn_frame = randint(60, 150)
    clock.tick(60)
    frame_counter += 1
    score_timer += 2
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    # Increase difficulty every 600 frames (~10 seconds at 60 FPS)
    if frame_counter % 600 == 0:
        speed_level = min(speed_level + 1, 100)
        for e in enemies:
            e.vel = speed_level

    all_dead = True
    for agent in agents:
        if score_timer % 180 == 0 and agent.alive:
            agent.fitness += 5
        if agent.fitness > epoch_best_score:
            epoch_best_score = agent.fitness
        for g in enemies:
            agent.update(g)
            agent.check_collision(g)
        if agent.alive:
            all_dead = False

    redraw_game_window(agents, enemies, generation, fps, best_score, epoch_best_score)

    if all_dead:
        speed_level = 9  # Reset speed at the beginning of each epoch
        best_agent = max(agents, key=lambda a: a.fitness)
        if best_agent.fitness > best_score:
            best_score = best_agent.fitness
        generation += 1
        agents = evolve_population(agents)
        fitness_plot_surface = draw_fitness_plot(best_fitness_history, avg_fitness_history)
        for i, agent in enumerate(agents):
            agent.player.y = 550  # reset y position to the floor
        enemies = increase_difficulty(generation)

pygame.quit()

