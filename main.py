import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers, models


def CNN_Predict(img):
    (training_images,training_labels) ,(testing_images,testing_labels) = datasets.cifar10.load_data()
    training_images, testing_images = training_images /255 ,testing_images /255

    class_names = ['Paper','Stone','Metal','Pencil','Pen','Cloth']
    de_gradlist=[]
    non_de_gradlist =[]
    metal_list=[]

    # for i in range(16):
    #     plt.subplot(4,4,i+1)
    #     plt.imshow(training_images[i],cmap=plt.cm.binary)
    #     plt.xlabel(training_labels[i][0])


    # plt.show()

    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]

    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape = (32,32,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation = "relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(10,activation="softmax"))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))

    loss, accuracy = model.evaluate(testing_images,testing_labels)
    print(f"loss : {loss}")
    print(f"accuracy :{accuracy}")

    model.save('waste_classifier.model')

    img = cv.imread('pen.jpg')
    img  = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    plt.imshow(img,cmap=plt.cm.binary)
    prediction = model.predict(np.array([img])/255)
    index = np.argmax(prediction)
    print(prediction[index])



img = cv.VideoCapture(0)
img_counter = 0
while(True):

    ret, frame = img.read()
    cv.imshow('frame', frame)
    if key%255 == 27:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name,frame)
        print("{}written!".format(img_name))
        img_counter += 1
    img.release()
    img.destroyAllWindows()


