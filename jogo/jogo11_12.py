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
POPULATION_SIZE = 50
MUTATION_RATE = 0.1

# Track fitness history for plotting
best_fitness_history = []
avg_fitness_history = []

# Enemy colors (geometric shapes)
ENEMY_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Genome persistence
BEST_GENOME_FILE = "best_genome.npy"

def save_best_genome(genome):
    np.save(BEST_GENOME_FILE, genome)

def load_best_genome():
    if os.path.exists(BEST_GENOME_FILE):
        return np.load(BEST_GENOME_FILE)
    return None

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
    def __init__(self, x, width=40, height=60, speed=9):
        y = 510 # Adjusted from 480 → now right at head level
        
        #optional
        #y = randint(410, 430)  # Still within dangerous range for head

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
        self.brain = NeuralNetwork(4, 6, 2, genome)
        self.fitness = 0
        self.alive = True
        self.last_inputs = None
        self.last_hidden = None

    def update(self, obstacle):
        if not self.alive:
            return

        # Inputs for the neural network
        dist = obstacle.x - self.player.x
        height = obstacle.y
        speed = obstacle.vel
        obstacle_type = 1 if isinstance(obstacle, FlyingEnemy) else 0

        inputs = np.array([dist / 900, height / 800, speed / 10, obstacle_type])
        h, decision = self.brain.forward(inputs)

        self.last_inputs = inputs
        self.last_hidden = h

        # UNPACK OUTPUTS
        jump_decision = decision[0]
        duck_decision = decision[1]

        # Handle jump
        if jump_decision > 0 and not self.player.is_jump:
            self.player.is_jump = True
            self.player.vertical_velocity = self.player.jump_velocity

        # Handle ducking
        self.player.is_ducking = duck_decision > 0 and not self.player.is_jump

        # Apply gravity and vertical movement
        if self.player.is_jump:
            self.player.vertical_velocity += self.player.gravity
            self.player.y += self.player.vertical_velocity

            if self.player.y >= 550:
                self.player.y = 550
                self.player.is_jump = False
                self.player.vertical_velocity = 0

        self.fitness += 1

    def check_collision(self, obstacle):
        dino_rect = pygame.Rect(self.player.box)
        enemy_rect = pygame.Rect(obstacle.box)
        if dino_rect.colliderect(enemy_rect):
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
        self.is_ducking = False
        self.duck_img = pygame.image.load('boneco_duck.png')  # You’ll need to create or find a duck sprite
        self.vertical_velocity = 0
        self.jump_velocity = -15
        self.gravity = 0.5
        self.char_img = pygame.image.load('boneco.png')
        self.box = (self.x, self.y, 100, 150)

    def reset(self):
        self.y = 550
        self.vertical_velocity = 0
        self.is_jump = False
        self.is_ducking = False

    def draw(self, win):
        if self.is_ducking:
            win.blit(self.duck_img, (self.x, self.y + 40))  # Adjust position when ducking
            self.box = (self.x + 10, self.y + 40, 100, 110)  # Shorter hitbox
        else:
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

# Add a surface plot function to show fitness evolution
plot_surface = pygame.Surface((400, 350))
def draw_fitness_plot():
    global plot_surface
    fig = plt.figure(figsize=[5, 3])
    plt.plot(best_fitness_history[-40:], label="Best", color='blue')
    plt.plot(avg_fitness_history[-40:], label="Average", color='orange')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    plot_surface = surf.copy()
    plt.close(fig)

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

    # Draw connections from input to hidden
    for i in range(nn.input_size):
        for j in range(nn.hidden_size):
            color = (255, 0, 0) if nn.w1[i][j] > 0 else (0, 0, 255)
            pygame.draw.line(win, color,
                             (start_x, start_y + i * spacing_y),
                             (start_x + 60, start_y + j * spacing_y), 1)

    # Output neurons: two (jump and duck)
    output_labels = ["Jump", "Duck"]
    output_spacing = 30  # spacing between output neurons

    for k in range(2):  # nn.output_size is 2
        y_offset = start_y + (k + 2) * spacing_y  # position output neurons lower
        pygame.draw.circle(win, (0, 255, 0), (start_x + 120, y_offset), 10)
        win.blit(font2.render(output_labels[k], True, (0, 0, 0)), (start_x + 135, y_offset - 10))

        for j in range(nn.hidden_size):
            color = (255, 0, 0) if nn.w2[j][k] > 0 else (0, 0, 255)
            pygame.draw.line(win, color,
                             (start_x + 60, start_y + j * spacing_y),
                             (start_x + 120, y_offset), 1)


def evolve_population(old_agents):
    sorted_agents = sorted(old_agents, key=lambda a: a.fitness, reverse=True)
    best = sorted_agents[0].fitness
    avg = sum(a.fitness for a in old_agents) / len(old_agents)
    best_fitness_history.append(best)
    avg_fitness_history.append(avg)
    top_genomes = [a.brain.genome for a in sorted_agents[:5]]
    save_best_genome(sorted_agents[0].brain.genome)
    draw_fitness_plot()
    return [DinoAgent(10 + i * 5, 550, 64, 64, mutate(choice(top_genomes))) for i in range(POPULATION_SIZE)]

def mutate(genome):
    return [gene + uniform(-0.5, 0.5) if uniform(0, 1) < MUTATION_RATE else gene for gene in genome]

def redraw_game_window(agents, enemies, generation, fps, best_score, epoch_best_score):
    draw_background(win)
    alive = sum(agent.alive for agent in agents)
    win.blit(font.render(f'Generation: {generation}', 1, (0, 0, 0)), (10, 10))
    win.blit(font.render(f'Alive: {alive}', 1, (0, 0, 0)), (10, 30))
    win.blit(font.render(f'FPS: {int(fps)}', 1, (0, 0, 0)), (10, 50))
    win.blit(font.render(f'Best Score (All Time): {best_score}', 1, (0, 0, 0)), (10, 70))
    win.blit(font.render(f'Best Score (This Epoch): {epoch_best_score}', 1, (0, 0, 0)), (10, 90))
    win.blit(font.render(f'Speed: {speed_level}', 1, (0, 0, 0)), (10, 110))

    if agents:
        win.blit(font.render(f'Jump Velocity: {agents[0].player.jump_velocity}', 1, (0, 0, 0)), (10, 130))
        win.blit(font.render(f'Gravity: {agents[0].player.gravity}', 1, (0, 0, 0)), (10, 150))
        win.blit(font.render(f'Vertical Velocity: {int(agents[0].player.vertical_velocity)}', 1, (0, 0, 0)), (10, 170))

    # Draw only best agent

    best = max(agents, key=lambda a: a.fitness)

    for agent in agents:
        if agent.alive:
            agent.draw(win)

    for obs in enemies[:]:
        if obs.active:
            obs.draw(win)
        else:
            enemies.remove(obs)

    if best.last_inputs is not None and best.last_hidden is not None:
        draw_network(best.brain, best.last_inputs, best.last_hidden, win)

    win.blit(plot_surface, (350, 10))  # Fitness chart
    pygame.display.update()

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
            x_pos += randint(40, 60)
            enemy2 = enemy(x_pos, 636, speed=speed_level)
            enemies.append(enemy2)
    for e in enemies:
        e.vel = speed
    return enemies

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
enemy_spawn_timer = 0
frame_counter = 0
next_enemy_spawn_frame = randint(90, 225)  # adjusted for 60 FPS
run = True
best_score = 0
speed_level = 9
epoch_best_score = 0
score_timer = 0
last_spawn_time = 0  # For controlling enemy spawn timing

# I join the jogo9 while run with de rest of the jogo11.py
# bellow part was get in jogo 9
while run:
    fps = clock.get_fps()
    enemy_spawn_timer += 1

    # Spawn new enemy during gameplay
    if enemy_spawn_timer >= next_enemy_spawn_frame:
        base_x = 1200
        is_flying = uniform(0, 1) < 0.3
        if is_flying:
            new_enemy = FlyingEnemy(base_x, width=40, height=60, speed=speed_level)
        else:
            new_enemy = enemy(base_x, 636, speed=speed_level, width=40, height=60)
        new_enemy.vel = speed_level
        enemies.append(new_enemy)

        # 25% chance to spawn second close enemy
        if uniform(0, 1) < 0.25:
            # If both are same type, use normal spacing
            if is_flying:
                spacing = randint(200, 300)
                second_enemy = FlyingEnemy(base_x + spacing, width=40, height=40, speed=speed_level)
            else:
                spacing = randint(40, 60)
                second_enemy = enemy(base_x + spacing, 636, speed=speed_level, width=40, height=60)

            # But if types would be different, use large spacing
            if uniform(0, 1) < 0.3:
                # Opposite type second enemy
                large_spacing = randint(300, 450)
                if is_flying:
                    second_enemy = enemy(base_x + large_spacing, 636, speed=speed_level, width=40, height=60)
                else:
                    second_enemy = FlyingEnemy(base_x + large_spacing, width=40, height=40, speed=speed_level)

            second_enemy.vel = speed_level
            enemies.append(second_enemy)

        enemy_spawn_timer = 0
        next_enemy_spawn_frame = randint(80, 150)

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
        epoch_best_score = 0  # 🔧 Reset epoch best score

        agents = evolve_population(agents)

        for agent in agents:
            agent.player.reset()

        enemies = []  # 🔧 Clear all enemies for the new epoch
        enemy_spawn_timer = 0  # 🔧 Reset spawn timer (optional, but recommended)
        next_enemy_spawn_frame = randint(90, 225)

pygame.quit()

