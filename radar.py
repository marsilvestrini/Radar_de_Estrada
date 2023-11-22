from __future__ import print_function
import cv2 as cv
from cap_from_youtube import cap_from_youtube
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time


backSub = cv.createBackgroundSubtractorKNN()
capture = cap_from_youtube('https://www.youtube.com/watch?v=nt3D26lrkho&ab_channel=VK', '720p')

cv.namedWindow('video', cv.WINDOW_NORMAL)

if not capture.isOpened():
    print('Unable to open')
    exit(0)

cont = 0
cxy_old = []
cxy_new = []

cy_new = []
cy_old = []

cont_carros = 0

while True:
    #leitura do frame
    ret, frame = capture.read()     
    frame2 = frame
    if frame is None:
        break
   
    #PRÉ PROCESSAMENTO

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    fgMask = backSub.apply(gray) 

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))

    closing = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    blur = cv.GaussianBlur(opening, (21, 21), 0) 
    retvalbin, bins = cv.threshold(blur, 220, 255, cv.THRESH_BINARY)  #<---------

    #ENCONTRANDO CONTORNOS
    contours, hierarchy = cv.findContours(bins[320:700,600:1100], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    hull = [cv.convexHull(c) for c in contours]

    #DESENHANDO CONTORNOS
    cv.drawContours(frame[320:700,600:1100], hull, -1, (0, 255, 0), 3)

    #LINHA PARA PARAR CONTAGEM
    lineypos = 150

    #LINHA PARA INICIAR CONTAGEM
    lineypos2 = 200
    cv.line(frame[320:700,600:1100], (0, lineypos2), (800, lineypos2), (0, 255, 0), 2)

    #AREA MINIMA DE CONTORNO
    minarea = 300

    #AREA MAXIMA DE CONTORNO
    maxarea = 50000

    #VETORES PARAR ARMAZENAR OS CENTROIDS
    cxx = np.zeros(len(contours))
    cyy = np.zeros(len(contours))

    #INICIANDO OS CONTORNOS
    for i in range(len(contours)):  

        if hierarchy[0, i, 3] == -1:  

            area = cv.contourArea(contours[i])  #AREA DE CONTORNO

            if minarea < area < maxarea:  

                #IDENTIFICAÇÃO DOS CONTORNOS
                cnt = contours[i]
                M = cv.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                if cy > lineypos2 and cy < lineypos2+10.5:
                    cont_carros += 1
    
                if cy > lineypos:  #DEFINE OS CENTROIDS ABAIXO DA LINHA (Y INICIA A CONTAGEM DE CIMA PRA BAIXO)

                    #PEGA OS PONTOS DO RETÂNGULO DE DEMARCAÇÃO
                    #X,Y SÃO O CANTO SUPERIOR ESQUERDO; W,H SÃO LARGURA E ALTURA
                    x, y, w, h = cv.boundingRect(cnt)
                    cx =  int(x + w/2)
                    cy =  int(y + h/2)

                    #CRIA O RETÂNGULO E REPRESENTA OS CENTROIDS
                    cv.rectangle(frame[320:700,600:1100], (x, y), (x + w, y + h), (255, 0, 0), 2)

                    cv.drawMarker(frame[320:700,600:1100], (cx, cy), (0, 0, 255), cv.MARKER_STAR, markerSize=5, thickness=1,
                                    line_type=cv.LINE_AA)
                    
                    
                    #ARMAZENA OS VALORES DE CENTROID
                    cxx[i] = cx
                    cyy[i] = cy



    #ELIMINA OS VALORES DE CENTROID NULOS
    cxx = cxx[cxx != 0]
    cyy = cyy[cyy != 0]

    #ARMAZENAMENTO DOS FRAMES EM LISTAS

    if len(cxy_old) > 0:
        cxy_old = cxy_new.copy()
        cxy_new = np.array([cxx,cyy])

    else: 
        cxy_old = np.array([cxx,cyy])
   

    #DEFININDO A RELAÇÃO METRO/PIXEL 
    # uma faixa = 115 pixels = 3,60 metros (devio aos caminhões passando na imagem)
    dist_pixel = 3.6/115

    #IDENTIFICANDO FPS
    fps = capture.get(cv.CAP_PROP_FPS)
    fps = int(fps)

    #IDENTIFICAÇÃO DOS PARES DE VETORES E CÁLCULO DA VELOCIDADE

    if np.sum(cxy_old) and np.sum(cxy_new) and len(cxy_old[0]) == len(cxy_new[0]):
        
        for i in range(len(cxy_old[0])):
            print(cxy_old) 
            #KNN
            pontos_old = np.array([cxy_old[0][i], cxy_old[1][i]]).reshape(1, -1)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(pontos_old)
            pontos_current = np.array([cxy_new[0][i], cxy_new[1][i]]).reshape(1, -1)
            distances, indices = nbrs.kneighbors(pontos_current)
            
            #DESENHA A LINHA ENTRE CENTROIDS USANDO A DISTÂNCIA ENTRE ELES COMO BASE
            coordenadas_x_old = int(cxy_old[0][i])
            coordenadas_y_old = int(cxy_old[1][i])
            coordenadas_x_new = int(cxy_new[0][i])
            coordenadas_y_new = int(cxy_new[1][i])
            
            #CALCULO DE VELOCIDADE 
            distances = np.linalg.norm(distances)
            velocidade = (distances*dist_pixel)/(1/fps)
            velocidade = velocidade * 3.6
            velocidade = np.round(velocidade,2)
            cv.putText(frame[320:700,600:1100], str(velocidade),(coordenadas_x_new,coordenadas_y_new), cv.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 0, 255), 2)

            
    cv.putText(frame[320:700,600:1100], "FPS: " + str(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, .5, (205, 0, 255), 2)
    cv.putText(frame[320:700,600:1100], "N carros: " + str(cont_carros), (0, 30), cv.FONT_HERSHEY_SIMPLEX, .5, (205, 0, 255), 2)

    res = cv.resize(frame[320:700,600:1100],None,fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC)

    cv.imshow("FRAME",res)
    cv.imshow("BINS", bins[320:700,600:1100])
    # cv.imshow('1',frame2)

    keyboard = cv.waitKey(30)


    if keyboard == 'q' or keyboard == 27:
        break