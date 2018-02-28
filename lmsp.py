#!/usr/bin/env python

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import random
import cv2
import numpy as np

def strip_bg(bg, img):
    c1,c2,c3,c4 = cv2.split(img)

    c1_inv, c2_inv, c3_inv  = np.where(c1 == bg[0], 1, 0), np.where(c2 == bg[1], 1, 0), np.where(c3 == bg[2], 1, 0)
    alpha_and = np.zeros(c1.shape, dtype=np.uint8)
    alpha_and = np.where(np.logical_and(c1_inv, c2_inv), 1, 0)
    alpha_and = np.where(np.logical_and(alpha_and, c3_inv), 1, 0)
    new_alpha = np.where(alpha_and, 0, 255)
    new_alpha = np.array(new_alpha, dtype=np.uint8)

    #print new_alpha, "\n\n", c1
    return cv2.merge((c1, c2, c3, new_alpha))


class Cube(object):
    distance = 0
    left_key = False
    right_key = False
    up_key = False
    down_key = False
    a_key = False
    s_key = False
    d_key = False
    r_key = False
    f_key = False
    x_axis = 0
    y_axis = 0
    vertices = (
        (1,-1,-2),
        (1,-1,2),
        (-1,-1,2),
        (-1,-1,-2),
        (1,1,-2),
        (1,1,2),
        (-1,1,2),
        (-1,1,-2)
        )
    faces = (
        (1,2,3,4)
  #      (5,8,7,6),
  #      (1,5,6,2)
  #      (2,6,7,3)
  #      (3,7,8,4)
  #      (5,1,4,8)
        )
    edges = (
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 7),
        (6, 3),
        (6, 4),
        (6, 7),
        (5, 1),
        (5, 4),
        (5, 7)
    )
    texcoord = ((0,0),(1,0),(1,1),(0,1))
    #-------------------------------------
    def __init__(self, texture):
        self.coordinates = [0,0,0]
        self.rubik_id = self.load_texture(texture)

    def load_texture(self,filename):
        textureSurface = pygame.image.load(filename)
        textureData = pygame.image.tostring(textureSurface,"RGBA",1)
        width = textureSurface.get_width()
        height = textureSurface.get_height()
        ID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D,ID)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,textureData)
        return ID

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glTranslatef(0,0,-5)   
        
        glTranslatef(self.x_axis,self.y_axis,self.distance)
        
        glRotatef(self.coordinates[0],1,0,0)
        glRotatef(self.coordinates[1],0,1,0)
        glRotatef(self.coordinates[2],0,0,1)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D,self.rubik_id)
        
        glBegin(GL_QUADS)
        
        for i,v in enumerate(self.faces):
            glTexCoord2fv(self.texcoord[i])
            glVertex3fv(self.vertices[v -1])
        glEnd()
        glDisable(GL_TEXTURE_2D)
        """
        glBegin(GL_LINES)
        for edge in self.edges:
            glColor3fv((0,0,0))
            for vertex in edge:
                glVertex3fv(self.vertices[vertex])
        glEnd()
        """


    def rotate_x(self):
        if self.coordinates[0] > 360:
            self.coordinates[0] = 0
        else:
            self.coordinates[0] += 2
            
    def rotate_y(self):
        if self.coordinates[1] > 360:
            self.coordinates[1] = 0
        else:
            self.coordinates[1] += 2
            
    def rotate_z(self):
        if self.coordinates[2] > 360:
            self.coordinates[2] = 0
        else:
            self.coordinates[2] += 2
            
    def move_away(self):
        self.distance -= 0.1
        
    def move_close(self):
        if self.distance < 0:
            self.distance += 0.1
            
    def move_left(self):
        self.x_axis -= 0.1
        
    def move_right(self):
        self.x_axis += 0.1
        
    def move_up(self):
        self.y_axis += 0.1
        
    def move_down(self):
        self.y_axis -= 0.1
            
    def keydown(self):
        if self.a_key:
            self.rotate_x()
        elif self.s_key:
            self.rotate_y()
        elif self.d_key:
            self.rotate_y()
        elif self.r_key:
            self.move_away()
        elif self.f_key:
            self.move_close()
        elif self.left_key:
            self.move_left()
        elif self.right_key:
            self.move_right()
        elif self.up_key:
            self.move_up()
        elif self.down_key:
            self.move_down()
            
    def keyup(self):
        self.left_key = False
        self.right_key = False
        self.up_key = False
        self.down_key = False
        self.a_key = False
        self.s_key = False
        self.d_key = False
        self.r_key = False
        self.f_key = False
    
    def delete_texture(self):
        glDeleteTextures(self.rubik_id)
    
def main(label, texture):
    pygame.init()
    window = pygame.display.set_mode((640,480),pygame.DOUBLEBUF|pygame.OPENGL)

    pygame.display.set_caption("PyOpenGL Tutorial")
    clock = pygame.time.Clock()
    done = False
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(45,640.0/480.0,0.1,200.0)
    
    glEnable(GL_DEPTH_TEST)

    cube = Cube(texture)
    for i in range(45):
        cube.rotate_y()
    #----------- Main Program Loop -------------------------------------
    # --- Main event loop
    for i in range(0,700): # User did something
        #glClearColor(random.random(), random.random(), random.random(), 0)
        glClearColor(1, 1, 1, 1)

        if i % 3 == 0:
            r1 = random.randint(0,300)
            for q in range(r1):
                cube.rotate_x()
                pass
        elif i % 2 == 0:
            r1 = random.randint(0,300)
            for q in range(r1):
                cube.rotate_y()
                pass                
        elif i % 3 == 0:
            r1 = random.randint(0,300)
            for q in range(r1):
                #cube.rotate_z()
                pass
        #time.sleep(1)
        #cube.keydown()
        #window.blit(image, (0, 0))
        cube.render_scene()

        pygame.display.flip()
        #clock.tick(30)
        pygame.image.save(window, 'gates/' + str(label) + "_image" + str(i) + ".png")
        img = cv2.imread('gates/' + str(label) + "_image" + str(i) + ".png", cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        #print img.shape
        bg = (255,255,255)
        cv2.imwrite('gates/' + str(label) + "_image" + str(i) + ".png", strip_bg(bg, img))
    
    cube.delete_texture()
    pygame.quit()

if __name__ == '__main__':
    i = 'gates'
    q = 6
    main(i,"base_images/gate_trans.png")
