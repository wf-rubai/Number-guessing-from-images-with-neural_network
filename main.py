import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from read_idx1 import read_idx1_ubyte
from read_idx3 import read_idx3_ubyte
from NeuralNetwork import NeuralNetwork
from pygame.locals import*

pg.init()

images = read_idx3_ubyte('neural_network/train-images.idx3-ubyte')
labels = read_idx1_ubyte('neural_network/train-labels.idx1-ubyte')
test_images = read_idx3_ubyte('neural_network/t10k-images.idx3-ubyte')
test_labels = read_idx1_ubyte('neural_network/t10k-labels.idx1-ubyte')

images = images/255.0
test_images = test_images/255.0

images = images.reshape(images.shape[0],-1)
test_images = test_images.reshape(test_images.shape[0],-1)

nn = NeuralNetwork(784, 30, 10)
nn.train(images, labels, 10)

predictions = np.argmax(nn.predict(test_images), axis=1)
accuracy = np.mean(predictions == test_labels)
print(f'Test accuracy: {accuracy}')

fig, ax = plt.subplots(figsize=(1.75,.7), dpi=100)
prob_data = [.1, .05,.3,.7,.2,.5,.07,.09,.1,0]
prob_data.clear()

ax.plot(prob_data, marker='.', color='#3856ff', mec='#8996e0', mfc='#8996e0', ms=3, linewidth=.8)
ax.set_facecolor('#343445')
fig.patch.set_facecolor('#343445')
ax.grid(linewidth=.1)

pos = ax.get_position()
shift_x = 25 / fig.dpi / fig.get_figwidth()
shift_y = 10 / fig.dpi / fig.get_figheight()
new_pos = [pos.x0 + shift_x, pos.y0 + shift_y, pos.width, pos.height]
ax.set_position(new_pos)

ax.set_ylabel('Probability', fontsize = 5, color='#a6a6ad')
ax.spines['top'].set_color(None)
ax.spines['bottom'].set_color('#a6a6ad')
ax.spines['left'].set_color('#a6a6ad')
ax.spines['right'].set_color(None)
ax.spines['left'].set_gid(.5)
ax.tick_params(axis='both', which='major', labelsize=3, color='#828282', labelcolor = '#d9d9d9',length=2)

canvas = FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()
size = canvas.get_width_height()
surf = pg.image.fromstring(raw_data, size, "RGB")

width, hight = 700, 300
scale = 10
dmt = 28
margin = 10
scr = pg.display.set_mode((width, hight))
img = np.zeros((dmt, dmt))
font1 = pg.font.Font(None,25)
font2 = pg.font.Font(None,35)
text = font1.render("Prediction :", True, (146, 146, 150))
text2 = font1.render("Confidency :", True, (146, 146, 150))
predtext = font1.render("", True, (146, 146, 150))
predtext1 = font1.render("", True, (146, 146, 150))

run = True
while run:
    scr.fill((52, 52, 69))
    scr.blit(surf,(325,10))
    scr.blit(text,(350,210))
    scr.blit(text2,(350,240))
    scr.blit(predtext,(460, 210))
    scr.blit(predtext1,(460, 240))

    for ev in pg.event.get():
        if ev.type == QUIT:
            run = False
        elif ev.type == KEYDOWN:
            if ev.key == K_BACKSPACE:
                img.fill(0)
            elif ev.key == K_RETURN:
                a = img
                a = a.flatten()
                predictions = nn.predict(a)
                index = np.argmax(predictions)
                prob_data = predictions[0]
                # print(index)
                # print(predictions)
                ax.clear()  # Clear previous plot data
                ax.plot(prob_data, marker='.', color='#3856ff', mec='#8996e0', mfc='#8996e0', ms=3, linewidth=.8)
                ax.grid(linewidth=.1)

                canvas.draw()
                raw_data = renderer.tostring_rgb()
                surf = pg.image.fromstring(raw_data, size, "RGB")

                predtext = font2.render(f"{index}", True, (215, 215, 217))
                predtext1 = font2.render(f"{(predictions[0][index]*100):.2f}%", True, (215, 215, 217))
        elif ev.type == MOUSEMOTION and pg.mouse.get_pressed()[0]:
            x, y = ev.pos
            img[(y-margin)//scale+1][(x-margin)//scale] = .5
            img[(y-margin)//scale-1][(x-margin)//scale] = .5
            img[(y-margin)//scale][(x-margin)//scale] = 1

    for i in range(dmt):
        for j in range(dmt):
            a = 255*img[i][j]
            con = 1
            pg.draw.rect(scr, (a,a,a), (margin+(j*scale),margin+(i*scale), scale, scale))

    pg.time.delay(10)    
    pg.display.flip()

pg.quit()
    