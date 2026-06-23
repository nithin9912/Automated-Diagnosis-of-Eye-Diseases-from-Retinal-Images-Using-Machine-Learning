from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from skimage import io, transform
from sklearn import preprocessing
import numpy as np
import joblib
import cv2
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pandas as pd

main=tkinter.Tk()
main.title("Automated Diagnosis of Eye Diseases from Retinal Images Using Machine Learning ")
main.geometry("1300x1200")
title=tkinter.Label(main,text=" Automated Diagnosis of Eye Diseases from Retinal Images Using Machine Learning",justify="center")
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')
title.config(bg="magenta4", fg='thistle1')  
title.config(font=ff)           
title.config(height=3, width=120)       
title.place(x=70,y=10)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')
global filename
global accuracy, graph
X = []
Y = []
global classifier
categories = ['glaucoma','normal']


global X_train,y_train,X_test,y_test

model_folder = "model"


def uploadDataset(): #function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    categories


def Preprocessing():
    global model_folder,X,Y
    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")
    if os.path.exists(X_file) and os.path.exists(Y_file):
      X = np.load(X_file)
      Y = np.load(Y_file)
      print("X and Y arrays loaded successfully.")
      text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")

    else:
      X = [] # input array
      Y = [] # output array
      for root, dirs, directory in os.walk(filename):
          for j in range(len(directory)):
              name = os.path.basename(root)
              print(f'Loading category: {dirs}')
              print(name+" "+root+"/"+directory[j])
              if 'Thumbs.db' not in directory[j]:
                  img_array = cv2.imread(root+"/"+directory[j])
                  img_resized = resize(img_array, (64, 64, 3))
                # Append the input image array to X
                  X.append(img_resized.flatten())
                # Append the index of the category in categories list to Y
                  Y.append(categories.index(name))
      X = np.array(X)
      Y = np.array(Y)
      np.save(X_file, X)
      np.save(Y_file, Y)
      text.insert(tkinter.END,X.shape)
      text.insert(tkinter.END,Y.shape)

def splitting():
   global X_train,y_train,X_test,y_test
   X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=77)  
   text.insert(END,"X_train"+str(X_train.shape)+"\n\n")  
   text.insert(END,"Y_train"+str(y_train.shape)+"\n\n")  
   text.insert(END,"X_test"+str(X_test.shape)+"\n\n")  
   text.insert(END,"Y_test"+str(y_test.shape)+"\n\n")     

 
#defining global variables to store accuracy and other metrics
labels=categories
precision = []
recall = []
fscore = []
accuracy = []


#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' FSCORE      : '+str(f))
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n\n")
    report=classification_report(predict, testY,target_names=categories)
    print('\n',algorithm+" classification report\n",report)
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.show()


def dt():
   # Check if the pkl file exists
    Model_file = os.path.join(model_folder, "DT_Model.pkl")
    if os.path.exists(Model_file):
    # Load the model from the pkl file
        dt_classifier = joblib.load(Model_file)
        predict = dt_classifier.predict(X_test)
        calculateMetrics("DecisionTreeClassifier", predict, y_test)
    else:
    # Create Random Forest Classifier with Decision Tree as base estimator
        dt_classifier = DecisionTreeClassifier()
        dt_classifier.fit(X_train, y_train)
    # Save the model weights to a pkl file
        joblib.dump(dt_classifier, Model_file)  
        predict = dt_classifier.predict(X_test)
        print("Decision Tree Classifier model trained and model weights saved.")
        calculateMetrics("DecisionTreeClassifier", predict, y_test)

def rf():
    global rf_classifier
    # Check if the pkl file exists
    Model_file = os.path.join(model_folder, "RFC_Model.pkl")
    if os.path.exists(Model_file):
    # Load the model from the pkl file
        rf_classifier = joblib.load(Model_file)
        predict = rf_classifier.predict(X_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
    else:
    # Create Random Forest Classifier with Decision Tree as base estimator
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
    # Save the model weights to a pkl file
        joblib.dump(rf_classifier, Model_file)  
        predict = rf_classifier.predict(X_test)
        print("Random Forest model trained and model weights saved.")
        calculateMetrics("RandomForestClassifier", predict, y_test)


def pred():
    filename = filedialog.askopenfilename(initialdir="testImages")
    img=imread(filename)
    img_resize=resize(img,(64,64,3))
    img_preprocessed=[img_resize.flatten()]
    output_number=rf_classifier.predict(img_preprocessed)[0]
    output_name=labels[output_number]

    plt.imshow(img)
    plt.text(10, 10, f'Predicted Output of RFC: {output_name}', color='white', fontsize=12, weight='bold', backgroundcolor='black')
    plt.axis('off')
    plt.show()
    


upload=Button(main,text="upload Dataset",command=uploadDataset,bg="goldenrod1",fg="black")
upload.place(x=50,y=550) 
upload.config(font=ff)


upload=Button(main,text="Preprocessing data",command=Preprocessing,bg="goldenrod1",fg="black")
upload.place(x=400,y=550) 
upload.config(font=ff)

upload=Button(main,text="splitting",command=splitting,bg="goldenrod1",fg="black")
upload.place(x=750,y=550) 
upload.config(font=ff)

upload=Button(main,text="DecisionTree Classifier",command=dt,bg="goldenrod1",fg="black")
upload.place(x=50,y=600) 
upload.config(font=ff)





upload=Button(main,text="Random Forest Classifier",command=rf,bg="goldenrod1",fg="black")
upload.place(x=400,y=600) 
upload.config(font=ff)



p= Button(main, text="Prediction", command=pred,bg="goldenrod1",fg="black")
p.place(x=750, y=600)
p.config(font=ff)



text=Text(main,height=20,width=150,bg="azure")
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=ff)
main.mainloop()    
    
