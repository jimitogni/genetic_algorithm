import pygame
from random import randint

pygame.init()

clock = pygame.time.Clock()

# other vas
vel_obj = 10
gravity = 10
pos_x = 1010
pos_y = 380

# images
bg = pygame.image.load('fundo.png')

char_left = pygame.image.load('bonecol.png')
char_right = pygame.image.load('boneco.png')
char = pygame.image.load('boneco.png')

win = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Darwin Algoritimo Evolutivo")


class player(object):
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


class enemy(object):
    walk_left = [pygame.image.load('gato.png'), pygame.image.load('dog.png'), pygame.image.load('gir.png'), pygame.image.load('rato.png')]


def redraw_game_window():
    win.blit(bg, (0, 0))  # back ground image
    man.draw(win)
    pygame.display.update()


man = player(300, 400, 64, 64)
run = True

while run:

    clock.tick(40)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and man.x > man.vel:
        man.x -= man.vel
        man.left = True
        man.right = False
    elif keys[pygame.K_RIGHT] and man.x < 900 - man.width - man.vel:
        man.x += man.vel
        man.left = False
        man.right = True
    else:
        man.right = False
        man.left = False
        man.walk_count = 0

    if not man.is_jump:
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            man.is_jump = True
            man.right = False
            man.left = False
            man.walk_count = 0
    else:
        if man.jump_count >= -10:
            neg = 1
            if man.jump_count < 0:
                neg = -1
            man.y -= (man.jump_count ** 2) * 0.5 * neg
            man.jump_count -= 1
        else:
            man.is_jump = False
            man.jump_count = 10

    redraw_game_window()

# pos_x -= velocidade_objetos

# janela.blit(fundo, (0,0))
# janela.blit(boneco, (x,y))
# janela.blit(gato, (pos_x+ 200, pos_y))
# janela.blit(dog, (pos_x+ 300, pos_y))
# janela.blit(gir, (pos_x+ 400, pos_y))
# janela.blit(rato, (pos_x+ 500, pos_y))

pygame.quit()
