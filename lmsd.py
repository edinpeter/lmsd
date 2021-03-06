#!/usr/bin/env python

import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import random
import cv2
import numpy as np

strip = True
square_off = True
differ_sizes = True
PADDING = 10
differ_center = True
def strip_bg(bg, img):
    c1,c2,c3,c4 = cv2.split(img)

    c1_inv, c2_inv, c3_inv  = np.where(c1 == bg[0], 1, 0), np.where(c2 == bg[1], 1, 0), np.where(c3 == bg[2], 1, 0)
    alpha_and = np.zeros(c1.shape, dtype=np.uint8)
    alpha_and = np.where(np.logical_and(c1_inv, c2_inv), 1, 0)
    alpha_and = np.where(np.logical_and(alpha_and, c3_inv), 1, 0)
    new_alpha = np.where(alpha_and, 0, 255)
    new_alpha = np.array(new_alpha, dtype=np.uint8)

    return cv2.merge((c1, c2, c3, new_alpha))

def crop(img, min_thresh=np.array((0,0,0,255)), max_thresh=np.array((255,255,255,255))):
    mask = cv2.inRange(img,min_thresh, max_thresh)
    #mask = np.where(mask == 0, 255, 0)
    im2, contours,hierarchy = cv2.findContours(mask, 1, 2)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3)

    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    epsilon = 0.01 * perimeter
    bb_c = cv2.approxPolyDP(cnt, epsilon, True)
    min_x, max_x, min_y, max_y = get_rect(bb_c)
    #cv2.drawContours(img, [bb_c], -1, (0,0,255), 3)

    #cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0,255,0))
    max_dim = max(max_x - min_x, max_y - min_y)
    #print "Max dim: %i" % (max_dim)
#   cropped_img = img[min_y - PADDING : max_y + PADDING, min_x - PADDING : max_x + PADDING]
    cropped_img = img[min_x - PADDING : max_x + PADDING, min_y - PADDING : max_y + PADDING]
    #cv2.imshow("win", cropped_img)
    #cv2.waitKey(0)
    cropped_img = cv2.resize(cropped_img, (400,400))
    return cropped_img

def get_rect(bounding_box):
    min_x = 99999
    min_y = 99999
    max_y = -1
    max_x = -1
    for point in bounding_box:
        #print point
        min_x = min(min_x, point[0][1])
        max_x = max(max_x, point[0][1])
        max_y = max(max_y, point[0][0])
        min_y = min(min_y, point[0][0])
    return min_x, max_x, min_y, max_y


def square(image):
	height = image.shape[0]
	width = image.shape[1]

	square_img = image[ (height - 400) / 2.0 : height - (height - 400) / 2.0, (width - 400) / 2.0 : width - (width - 400) / 2.0]
	square_img = cv2.resize(square_img, (400,400))
	return square_img


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
        (1,-1,-1),
        (1,-1,1),
        (-1,-1,1),
        (-1,-1,-1),
        (1,1,-1),
        (1,1,1),
        (-1,1,1),
        (-1,1,-1)
        )
    faces = (
        (1,2,3,4),
        (5,8,7,6),
        (1,5,6,2),
        (2,6,7,3),
        (3,7,8,4),
        (5,1,4,8)
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
        
        for face in self.faces:
            for i,v in enumerate(face):
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

    def reset_distance(self):
    	self.distance = 0
    
    def reset_center(self):
    	self.x_axis = 0
    	self.y_axis = 0

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
    window_width = 640.0
    window_height = 480.0
    window = pygame.display.set_mode((int(window_width), int(window_height)),pygame.DOUBLEBUF|pygame.OPENGL)

    errors = 0

    pygame.display.set_caption("PyOpenGL Tutorial")
    clock = pygame.time.Clock()
    done = False
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(45, window_width / window_height, 0.1, 200.0)
    
    glEnable(GL_DEPTH_TEST)

    cube = Cube(texture)
    #----------- Main Program Loop -------------------------------------
    # --- Main event loop
    for i in range(0,2000): # User did something
        colors = (random.random(), random.random(), random.random(),)
        glClearColor(colors[0], colors[1], colors[2], 1)
        cube.reset_distance()
        cube.reset_center()

        ops = [cube.rotate_x, cube.rotate_y, cube.rotate_z]
        if differ_sizes:
        	ops = [cube.rotate_x, cube.rotate_y, cube.rotate_z, cube.move_away, cube.move_close]
       	shifts = [cube.move_up, cube.move_down, cube.move_right, cube.move_left]

       	if differ_center:
       		ops = ops + shifts
        for op in ops:
            r1 = random.randint(0,3) if op in shifts else random.randint(0,300) 
            for q in range(r1):
                op()



        cube.render_scene()

        pygame.display.flip()
        pygame.image.save(window, "dice/" + str(label) + "_image" + str(i) + ".png")
       
        if strip:
            img = cv2.imread('dice/' + str(label) + "_image" + str(i) + ".png", cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            bg = img[0][0]
            cv2.imwrite('dice_no_bg/' + str(label) + "_image" + str(i) + "_nbg.png", strip_bg(bg, img))
            cv2.imwrite('dice_squared/' + str(label) + "_image" + str(i) + "_squared.png", square(img))
            if square_off:
                img = cv2.imread('dice_no_bg/' + str(label) + "_image" + str(i) + "_nbg.png", cv2.IMREAD_UNCHANGED)
                #print img
                try:
                	cv2.imwrite('dice_no_bg_squared/' + str(label) + "_image" + str(i) + "_nbg.png", crop(img))
                except:
                	errors += 1
                	pass

    
    cube.delete_texture()
    pygame.quit()
    return errors

if __name__ == '__main__':
	errors = list()
	for i in range(1,7):
	   errors.append(main(i,"base_images/dice_" + str(i) + "_rgb.png"))
	print "Errors: ", errors
