import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tkinter import filedialog
# from PIL import Image, ImageTk
import PIL.Image
import PIL.ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tkinter import *
from tkinter import messagebox, ttk

class LCD_CNN:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Brain Tumor Detection")

        img4 = PIL.Image.open(r"xray/train/Tumor/Y55.jpg")
        img4 = img4.resize((1006, 500), PIL.Image.LANCZOS)
        self.photoimg4 = PIL.ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=50, width=1006, height=500)

        title_lbl = Label(text="Brain Tumor Detection", font=("Bradley Hand ITC", 30, "bold"), bg="black", fg="white",)
        title_lbl.place(x=0, y=0, width=1006, height=50)

        self.b1 = Button(text="Import Data", cursor="hand2", command=self.import_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b1.place(x=80, y=130, width=180, height=30)

        self.b3 = Button(text="Train Data", cursor="hand2", command=self.train_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b3.place(x=80, y=180, width=180, height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

        self.b4 = Button(text="Test Data", cursor="hand2", command=self.test_data, font=("Times New Roman", 15, "bold"), bg="white", fg="black")
        self.b4.place(x=80, y=230, width=180, height=30)
        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

    def import_data(self):
        self.dataDirectory = 'xray/train/'
        self.TumorPatients = os.listdir(self.dataDirectory)
        self.size = 10
        self.NoSlices = 5
        messagebox.showinfo("Import Data", "Data Imported Successfully!")

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        self.cnn = Sequential()

        self.cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Conv2D(32, (3, 3), activation="relu"))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Conv2D(32, (3, 3), activation="relu"))
        self.cnn.add(MaxPooling2D(pool_size=(2, 2)))
        self.cnn.add(Flatten())
        self.cnn.add(Dense(activation='relu', units=128))
        self.cnn.add(Dense(activation='sigmoid', units=1))

        self.cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory('xray/train', target_size=(64, 64), batch_size=32,     class_mode='binary')
        validation_generator = test_datagen.flow_from_directory('xray/val/', target_size=(64, 64), batch_size=32,   class_mode='binary')
        test_set = test_datagen.flow_from_directory('xray/test', target_size=(64, 64), batch_size=32,   class_mode='binary')

        cnn_model = self.cnn.fit(training_set, steps_per_epoch=8, epochs=9, validation_data=validation_generator,   validation_steps=20)

        test_accu = self.cnn.evaluate(test_set, steps=20)
        print('The testing accuracy is :', test_accu[1]*100, '%')
        messagebox.showinfo("ACCURACY", f"{test_accu[1]*100}%")
        messagebox.showinfo("Train Data", "Model Trained Successfully!")

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

    def test_data(self):
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
    
        img = PIL.Image.open(filename)
        plt.imshow(img)
        plt.show()
    
        img = load_img(filename, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = self.cnn.predict(x)
        print(classes)


        def ans():
            if classes > 0.5:
                return "Tumor"
            else:
                return "Normal"

        messagebox.showinfo("Test Data", ans())

if __name__ == "__main__":
    root = Tk()
    obj = LCD_CNN(root)
    root.mainloop()
