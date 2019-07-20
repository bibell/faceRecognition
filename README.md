# faceRecognition
face Recognition system using tkinter GUI  
When you execute finalregister.py file you will get complete graphical user interface
that makes you able to complete use of facial recognition system, 
their is actually four button is necessary the first button sayes detection demo
once you press it another window will appears than the Webcam began detect human face
Than it uses haarcascad classifier to classifies human faces. The second button 
is used to register new student that requires administrator user name and password, 
once the admin login another window will appears that have label and entry widgets
the labels are students information like name, ID and so on the entry's must be filled with
those information after you press register button, those information will goes in to sql database
and the Webcam will ready to capture student face, once human face appears in the Webcam it will 
capture the face 21 times than put it in the face database. The sql student information and 
face database is abviously used for train and recognition algorithms. The third button is used to train
The algorithms remember the more you train the face the better result you get(the better accurecy).
The fourth button is used to exit from the entaier applications. 

We design these project to show the possibilities that the paper base student ID card can be replaced with
Student natural face and afcourse it can be used for other access control that need identification
We did not use sofesticated algorithm as you see but we made these only to show possibilities 
That the old school system can be replaced with more advanced technology obviously implemented 
With advanced mechine learning algorithms as you see in the python code we didn't use more
Advanced mechin learning library like deep face and open face we only use opencv 
so we and you can do better. 
