# <h1 align="center">Radar de Estrada com Visão Computacional Clássica</h1>

Este projeto tem como objetivo simular um radar de estrada efetuando a contagem de carros passantes e o cálculo da velocidade instantânea a cada frame utilizando apenas ferramentas clássicas de visão computacional, ou seja, sem fazer uso de machine learning.

## Usabilidade</h2>

O projeto desenvolvido pode ser utilizado em qualquer video de rodovia no youtube para simular um radar de rodovia com registor de velocidade, basta substituir no código o link para o mesmo, não existe a necessidade de baixar o conteúdo.

## Bibliotecas Necessárias</h3>

Como mencionado, o projeto opera apenas com funções de visão computacional clássicas, então, serão necessárias apenas a biblioteca Opencv, e as bibliotecas numpy e sklear para cálculos matemáticos. A biblioteca cap_from_youtube serve a funcionalidade de utilizar um vídeo do youtube sem a necessidade de baixá-lo.

```python
import cv2 as cv
from cap_from_youtube import cap_from_youtube
import numpy as np
from sklearn.neighbors import NearestNeighbors
```

## Leitura do vídeo e Inicialização da Captura</h3>

```python

backSub = cv.createBackgroundSubtractorKNN()
capture = cap_from_youtube('https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK', '720p')

# cv.namedWindow('video', cv.WINDOW_NORMAL)

if not capture.isOpened():
    print('Unable to open')
    exit(0)
```
## Pré Processamento da Imagem</h3>

Nesta parte do código é feito tratamento da imagem a fim de encontrar uma máscara contendo as informações dos carros em movimento.

```python
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

fgMask = backSub.apply(gray) 

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

closing = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

blur = cv.GaussianBlur(opening, (21, 21), 0) 
retvalbin, bins = cv.threshold(blur, 220, 255, cv.THRESH_BINARY) 
```

## Localizando os Carros</h3>

A partir da máscara com os blobs dos carros é possíverl encontrar seu contorno para melhor localização de sua centróide.

```python
contours, hierarchy = cv.findContours(bins[320:700,600:1100], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

hull = [cv.convexHull(c) for c in contours]

```

## Localização de Centróides</h3>

Assim, a partir dos contornos, calcula-se a posição da centróide de cada carro individualmente na imagem dentro de um loop. A centróide é usada também para definir a quantidade de carros acima da linha pré definida para a contagem.

```python
cnt = contours[i]
M = cv.moments(cnt)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
```

## Cálculo da velocidade</h3>
a partir dos centroides de cada carro, faz-se um KNN (K-Nearest Neighbors) para identificar a distância entre os centróides mais próximos entre cada frame, a fim de calculoar a velocidade a partir de uma relação metro/pixel.

```python
pontos_old = np.array([cxy_old[0][i], cxy_old[1][i]]).reshape(1, -1)
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(pontos_old)
pontos_current = np.array([cxy_new[0][i], cxy_new[1][i]]).reshape(1, -1)
distances, indices = nbrs.kneighbors(pontos_current)

coordenadas_x_new = int(cxy_new[0][i])
coordenadas_y_new = int(cxy_new[1][i])

#CALCULO DE VELOCIDADE 
distances = np.linalg.norm(distances)
velocidade = (distances*dist_pixel)/(1/fps)
velocidade = velocidade * 3.6
velocidade = np.round(velocidade,2)
```

## Visualização de Resultados</h3>
Os resultados de contagem e velocidade podem ser visualizados no frame do vídeo.

```python
cv.putText(frame[320:700,600:1100], str(velocidade),(coordenadas_x_new,coordenadas_y_new), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv.putText(frame[320:700,600:1100], "N carros: " + str(cont_carros), (0, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (205, 0, 255), 2)
```
