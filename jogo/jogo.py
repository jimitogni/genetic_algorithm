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

win = pygame.display.set_mode((900, 800))
pygame.display.set_caption("Darwin Algoritimo Evolutivo")

score = 0

#classe do darwin
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
        self.box = (self.x, self.y, 100, 150)#caixa em volta

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

        self.box = (self.x + 10, self.y, 100, 150)#caixa em volta
        pygame.draw.rect(win, (255,0,0), self.box, 2)#caixa em volta

#obistaculos
class enemy(object):
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
        self.box = (self.x, self.y, 10, 50)#caixa em volta

    def draw(self, win):
        self.move()
        if self.walk_count + 1 <= 33:
            self.walk_count = 0

        if self.vel > 0:
            win.blit(self.walk_left[self.walk_count//3], (self.x, self.y))
            self.walk_count += 1
        else:
            win.blit(self.walk_left[self.walk_count//3], (self.x, self.y))
            self.walk_count += 1

        self.box = (self.x, self.y, 65, 84)#caixa em volta
        pygame.draw.rect(win, (255,0,0), self.box, 2)#caixa em volta


    def move(self):
        if self.x > 10*-1:
        #    if self.x + self.vel < self.path[0]:
            self.x -= self.vel
        else:
            self.x = 900
            #self.x -= self.vel
            #        self.vel = self.vel * -1
        #        self.walk_count = 0
        #else:
        #    if self.x - self.vel > self.path[1]:
        #        self.x += self.vel
        #    else:
        #        self.vel = self.vel * -1
        #        self.walk_count = 0

    def stop(self):
        self.x = 500
        self.vel = 0


def redraw_game_window():
    win.blit(bg, (0, 0))  # back ground image
    text = font.render('Pontuacao: ' + str(score), 1, (0,0,0))
    win.blit(text, (10, 30))

    man.draw(win)
    gobling.draw(win)
    pygame.display.update()

#man.box[0] == gobling.box[0] and man.box[1] < gobling.box[1] + gobling.box[3]:

# main looping
man = player(10, 350, 64, 64) #(x, y, largura, altura)
gobling = enemy(900, 400, 64, 64, 300) #(x, y, largura, altura, end)
run = True
font = pygame.font.SysFont('comicsans', 30, True)
font2 = pygame.font.SysFont('Arial', 21)

while run:

    clock.tick(40)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    #debug
    dbg1 = font2.render('Man X: ' + str(man.box[0]), 1, (255, 255, 0))
    win.blit(dbg1, (10, 610))

    dbg2 = font2.render('Man Y: ' + str(man.box[1]), 1, (255, 255, 0))
    win.blit(dbg2, (10, 630))

    dbg3 = font2.render('gobling X: ' + str(gobling.box[0]), 1, (255, 255, 0))
    win.blit(dbg3, (10, 650))

    dbg4 = font2.render('gobling Y: ' + str(gobling.box[1]), 1, (255, 255, 0))
    win.blit(dbg4, (10, 670))

    print("-"*20)
    print(f"man X: {man.box[0]} - Y : {man.box[1]}")
    print(f"gobling X: {gobling.box[0]} - Y : {gobling.box[1]}")
    print("-" * 20)

    #colisao
    for count in range(1):
    # se darwin.X for >= ao gobbling.X E darwin.Y =< gobling.Y + diferença de alturas
    # se darwin.X for >= ao gobbling.X E Ydarwin tem que ser > que Yvilao + alturada do box do vilao
    # man box self.box = (self.x, self.y, 100, 150)
    # gobling box self.box = (self.x, self.y, 65, 84)#caixa em volta
        if man.box[0] == gobling.box[0]: #and man.box[1] < gobling.box[1] + gobling.box[3]:
            man.vel = 0
            gobling.vel = 0
            print("------------------------")
            print("MOOOOREEEEEUUUUU")
            print('COLISÃO - PARAMETROS:')
            print('man.y - ', man.y)
            print('man.x - ', man.x)
            print('gobling x + 65 posição = ', gobling.box[0] + gobling.box[2])
            print('gobling y + 65 altura = ', gobling.box[1] + gobling.box[2])
            print("------------------------")
            gobling.stop()
            clock.tick(1)

            #print('man.box[0] - ', man.box[0])
            #print('gobling.box[0] - ', gobling.box[0])
            #print()
            #print('man.box[1] -', man.box[1])
            #print('gobling.box[1] -', gobling.box[1])
            #print('man.box[3] -', man.box[3])
            #print('gobling.box[3] -', gobling.box[3])
            print('\n')
        else:
            #print("------------------------")
            #print("PULOUUU")
            score += 0.001
        #count += 1

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
                man.y -= (man.jump_count ** 2) * 0.8 * neg
                man.jump_count -= 1
            else:
                man.is_jump = False
                man.jump_count = 10

        # colision
        if man.x >= gobling.box[3]+10:
            print('parametros PULANDO')
            print(man.box[2])
            print(gobling.box[3])
            print('\n')
            score += 1
        if man.x == gobling.box[3]:
            print('COLISÃO - PARAMETROS:')
            print('man.box[2] - ', man.box[2])
            print('gobling.box[3] - ', gobling.box[2])
            print('man.box[1] -', man.box[1])
            print('gobling.box[1] -', gobling.box[1])
            print('man.box[3] -', man.box[3])
            print('gobling.box[3] -', gobling.box[3])
            print('\n')

    redraw_game_window()

# pos_x -= velocidade_objetos

# janela.blit(fundo, (0,0))
# janela.blit(boneco, (x,y))
# janela.blit(gato, (pos_x+ 200, pos_y))
# janela.blit(dog, (pos_x+ 300, pos_y))
# janela.blit(gir, (pos_x+ 400, pos_y))
# janela.blit(rato, (pos_x+ 500, pos_y))

pygame.quit()
