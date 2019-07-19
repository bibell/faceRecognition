from Tkinter import *
import Tkinter as tk
from PIL import ImageTk,Image
import cv2
import time
import os
import sqlite3
import numpy as np
import pickle
import subprocess
import ttk
from ttk import Frame
import tkFileDialog
import tkMessageBox
import threading
#import pygame
##############################
#face recognize

def recognize():

	face_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('Classifiers/haarcascade_eye.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("trainer/training_data.yml")

	#To train using images captured or saved online
	#img = cv2.imread("him.jpg")

	def getProfile(Id):
	    conn=sqlite3.connect("MyData")
	    query="SELECT * FROM hu_data_base WHERE ID="+str(Id)
	    cursor=conn.execute(query)
	    profile=None
	    for row in cursor:
		profile=row
	    conn.close()
	    return profile




	faces=face_cascade.load('haarcascade_frontalface_default.xml')

	#to train using frames from video
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_COMPLEX
	while True:
	    #comment the next line and make sure the image being read is names img when using imread
	    ret, img = cap.read()
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	    faces = face_cascade.detectMultiScale(gray,1.3,5)
	    for (x,y,w,h) in faces:

		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

		# Hiding the eye detector for now
		# eyes = eye_cascade.detectMultiScale(roi_gray)
		# for (ex, ey, ew, eh) in eyes:
		#     cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
		nbr_predicted, conf = recognizer.predict(gray[y:y+h, x:x+w])
		if conf < 90:
		    profile=getProfile(nbr_predicted)
		    
		    #for cafterial use add some function that connects to the data base 
		    #if the person id exist than some kind of sound must be heard
	            #conn=sqlite3.connect("MyData")
                    #id=+id
                    #cursor=conn.execute("SELECT test FROM hu_data_base WHERE ID="+str(id))
                    #test=0
                    if profile != None:
		    #validation(test)  
			    #if profile != None:
				cv2.putText(img, "Name: "+str(profile[0]), (x, y+h+30), font, 0.4, (0, 0, 255), 1);
				cv2.putText(img, "IDS: " + str(profile[1]), (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
				cv2.putText(img, "department: " + str(profile[2]), (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
				#cv2.putText(img, "year: " + str(profile[3]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1); 
				cv2.putText(img, "entry: " + str(profile[4]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
		                #thread=threading.Thread(target=addone)
		                #thread.start()
                        #pygame.init()
                        #meu=pygame.mixer.Sound("mu.mp3")
                        #pygame.mixer.Sound.play(meu)   
                        #pygame.mixer.music.load("mu.mp3")

		        #hear add some function that write the id of the person in the database in increamented way
		        #to determain wither the person reapit or not
		    
		else:
		    cv2.putText(img, "Name: Unknown", (x, y + h + 30), font, 0.4, (0, 0, 255), 1);
		    cv2.putText(img, "id: Unknown", (x, y + h + 50), font, 0.4, (0, 0, 255), 1);
		    cv2.putText(img, "department: Unknown", (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
		    #cv2.putText(img, "year: Unknown", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
		    cv2.putText(img, "entry: Unknown", (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
                    #tkMessageBox.showinfo("face unknown","unauthorize person detected") 
                    #break 

	    cv2.imshow('img', img)
	    if(cv2.waitKey(1) == ord('q')):
		#there must be some kind of funcition over hear that elimenate the increamented id when we enter q
                addone()
		break

	cap.release()
	cv2.destroyAllWindows()







#################################################
#train the face


def traine():      
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	path = 'haramaya_students_dataSets'

	def getImagesWithID(path):
	    imagePaths=[os.path.join(path, f) for f in os.listdir(path)]
	    faces=[]
	    IDs=[]
	    for imagePath in imagePaths:
		faceImg = Image.open(imagePath).convert('L')
		faceNp = np.array(faceImg, 'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		IDs.append(ID)
		cv2.imshow('training', faceNp)
		cv2.waitKey(10)
	    return np.array(IDs), faces

	Ids, faces = getImagesWithID(path)
	recognizer.train(faces, Ids)

	if not os.path.exists('trainer'):
	    os.makedirs('trainer')

	recognizer.save('trainer/training_data.yml')
        cv2.destroyAllWindows() 
        tkMessageBox.showinfo("Training finished","the more you train the face\nthe batter accuracy you get")
	cv2.destroyAllWindows()              
	


#####################################################


#########################################################


def addone():

    conn=sqlite3.connect("MyData")
    cursor=conn.execute("SELECT test FROM hu_data_base WHERE ID="+str(id))
    #n=n+1
    conn.execute("UPDATE hu_data_base SET test="+str(1))
    conn.commit()
    conn.close()
 

######################################################

def addzero():

    conn=sqlite3.connect("MyData")
    cursor=conn.execute("SELECT test FROM hu_data_base WHERE ID="+str(id))
    #n=n+1
    conn.execute("UPDATE hu_data_base SET test="+str(0))
    conn.commit()
    conn.close()
 


#####################################################




#####################################################


def baba():
	  face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	  cam=cv2.VideoCapture(0)
	  
	  faces=face_cas.load('haarcascade_frontalface_default.xml')



	  sample_number=0

	  while True:
	       
	      chake,frame=cam.read()
	   
	      faces=face_cas.detectMultiScale(frame,
		                           scaleFactor=1.2,
		                           minNeighbors=5,
		                           minSize=(20,20))  

	      for (x,y,w,h) in faces:

		 cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
	       
	      cv2.imshow('face detection system',frame)
	      #key=cv2.waitKey(1)
	      if (cv2.waitKey(1)==ord('q')):
                                
		     break
                  
	  #time.sleep()
	  cam.release()
          #exit()
	  cv2.distroyAllwindows()

def logout():

       conn=sqlite3.connect("MyData")
       cursor=conn.execute("SELECT test FROM hu_data_base WHERE ID="+str(id))
       conn.execute("UPDATE hu_data_base SET test="+str(0))
       conn.commit()
       conn.close()
       exit() 
def database():
        #path='Home/Desktop/test/gui/combain'
        fil=tkFileDialog.askopenfilename()
        tkFileDialog.openfile(fil)


##########################################      
     
     

window=tk.Tk()
window.title("graphycal user interface")
window.geometry("1000x1000")

#global name
#global department
#global year
#global id


Fullname=StringVar()
id1=IntVar()
Department=StringVar()
Acadamicyear=StringVar()
password=StringVar()


name=Fullname.get()
id=id1.get()
department=Department.get()
year=Acadamicyear.get()
#pw=password.get(



Label(window,text="HARAMAYA INSTITUTE OF TECHNOLOGY",font=("",25,"bold"),bg="blue",fg="white").place(x=150,y=5)



path4="face.jpg"
img4=ImageTk.PhotoImage(Image.open(path4))

im4=Label(window,image=img4, height=400,width=850)
im4.place(x=10,y=50)


path2="face2.jpg"
img1=ImageTk.PhotoImage(Image.open(path2))

im1=Label(window,image=img1, height=600,width=900)
im1.place(x=20,y=40)


path3="woman.jpg"
img3=ImageTk.PhotoImage(Image.open(path3))

im3=Label(window,image=img3, height=200,width=850)
im3.place(x=80,y=40)


Label(window,text="Artificial Intelligence software\nface id",fg="black",font=("",25,"bold")).place(x=200,y=450)



be1=Button(window,text="detection demo", bg="black",fg="white",font=("roman",10), command=baba)
be1.place(x=100,y=600)

be1=Button(window,text="train the face", bg="black",fg="white",font=("roman",10), command=traine)
be1.place(x=300,y=600)

be1=Button(window,text="Recognize", bg="black",fg="white",font=("roman",10),command=recognize)
be1.place(x=630,y=600)

be1=Button(window,text="sign out", bg="black",fg="white",font=("roman",10),command=logout)
be1.place(x=800,y=600)

cafe=Button(window,text="tick",bg="black",fg="white",font=("",20,"bold"),command=addzero)
cafe.place(x=480,y=600)


window.mainloop()
