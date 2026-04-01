import dataset
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import cv2
import time
import urllib.request
start = time.time()
video = cv2.VideoCapture(0)
time.sleep(2)
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
pump = 21
pump1 = 20
m1=26
m2=19
m3=13
m4=6

GPIO.setup(pump,GPIO.OUT)
GPIO.setup(pump1,GPIO.OUT)
GPIO.setup(m1,GPIO.OUT)
GPIO.setup(m2,GPIO.OUT)
GPIO.setup(m3,GPIO.OUT)
GPIO.setup(m4,GPIO.OUT)

GPIO.output(pump,0)
GPIO.output(pump1,0)
GPIO.output(m1,1)
GPIO.output(m2,0)
GPIO.output(m3,1)
GPIO.output(m4,0)
time.sleep(4)
i=0
while(1):


    (grabbed, frame) = video.read()
    if not grabbed:
        break
    cv2.imshow("input", frame)
    i=i+1
    print(i)
    if (cv2.waitKey(1) & 0xFF == ord('q')) or i==100:
        GPIO.output(m1,0)
        GPIO.output(m2,0)
        GPIO.output(m3,0)
        GPIO.output(m4,0)
        time.sleep(1)
        cv2.imwrite('/home/pi/Desktop/leaf/data/test/test.jpg',frame)
        cv2.waitKey(1)
    
    # Path of  training images
        train_path = '/home/pi/Desktop/leaf/data/train'
        if not os.path.exists(train_path):
            print("No such directory")
            raise Exception
        # Path of testing images
        dir_path = '/home/pi/Desktop/leaf/data/test'
        if not os.path.exists(dir_path):
            print("No such directory")
            raise Exception
        
        # Walk though all testing images one by one
        for root, dirs, files in os.walk(dir_path):
            for name in files:

                print("")
                image_path = name
                filename = '/home/pi/Desktop/leaf/data/test/test.jpg'
                print(filename)
                image_size=128
                num_channels=3
                images = []
            
                if os.path.exists(filename):
                    
                    # Reading the image using OpenCV
                    image = cv2.imread(filename)
                    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                    images.append(image)
                    images = np.array(images, dtype=np.uint8)
                    images = images.astype('float32')
                    images = np.multiply(images, 1.0/255.0) 
                
                    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                    x_batch = images.reshape(1, image_size,image_size,num_channels)

                    # Let us restore the saved model 
                    sess = tf.Session()
                    # Step-1: Recreate the network graph. At this step only graph is created.
                    saver = tf.train.import_meta_graph('model/trained_model.meta')
                    # Step-2: Now let's load the weights saved using the restore method.
                    saver.restore(sess, tf.train.latest_checkpoint('./model/'))

                    # Accessing the default graph which we have restored
                    graph = tf.get_default_graph()

                    # Now, let's get hold of the op that we can be processed to get the output.
                    # In the original network y_pred is the tensor that is the prediction of the network
                    y_pred = graph.get_tensor_by_name("y_pred:0")

                    ## Let's feed the images to the input placeholders
                    x= graph.get_tensor_by_name("x:0") 
                    y_true = graph.get_tensor_by_name("y_true:0") 
                    y_test_images = np.zeros((1, len(os.listdir(train_path)))) 


                    # Creating the feed_dict that is required to be fed to calculate y_pred 
                    feed_dict_testing = {x: x_batch, y_true: y_test_images}
                    result=sess.run(y_pred, feed_dict=feed_dict_testing)
                    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                    print(result)

                    # Convert np.array to list
                    a = result[0].tolist()
                    r=0

                    # Finding the maximum of all outputs
                    max1 = max(a)
                    index1 = a.index(max1)
                    predicted_class = None

                    # Walk through directory to find the label of the predicted output
                    count = 0
                    for root, dirs, files in os.walk(train_path):
                        for name in dirs:
                            if count==index1:
                                predicted_class = name
                            count+=1

                    # If the maximum confidence output is largest of all by a big margin then
                    # print the class or else print a warning
                    for i in a:
                        if i!=max1:
                            if max1-i<i:
                                r=1                           
                    if r ==0:
                        if(predicted_class == 'Early_blight'):
                            predicted_class='healthy'
                        elif(predicted_class == 'healthy'):
                            predicted_class='Early_blight'
                            
                        print(predicted_class)
                        wp = urllib.request.urlopen("https://api.thingspeak.com/update?api_key=OHMQHOUQ63Q4WYQA&field1=" + str(predicted_class))
                        
                        if(predicted_class != 'healthy'):
                            if(predicted_class == 'Late_blight'):
                                print('for this you have to use prophylactic chemical 10ml')
                                print('for this you have to use mancozeb chemical 10ml')
                            if(predicted_class == 'Early_blight'):
                                print('for this you have to use ABC 10ml')
                                print('for this you have to use ABC chemical 10ml')

                            if(predicted_class == 'Bacterial_spot'):
                                print('for this you have to use copper chemical 10ml')
                                print('for this you have to use mancozeb chemical 10ml')

                            if(predicted_class == 'Leaf_Mold'):
                                print('for this you have to use cholothanil chemical 10ml')
                                print('for this you have to use maneb chemical 10ml')

                                
                            print('PUMP ON')
                            GPIO.output(pump,1)
                            GPIO.output(pump1,0)
                            time.sleep(5)
                            GPIO.output(pump,0)
                            GPIO.output(pump1,0)
                            print('PUMP OFF')

                            
                    else:
                        print("Could not classify with definite confidence")
                        print("Maybe:",predicted_class)
                    GPIO.output(m1,1)
                    GPIO.output(m2,0)
                    GPIO.output(m3,1)
                    GPIO.output(m4,0)
                    i=0
                # If file does not exist
                else:
                    print("File does not exist")
                
                    GPIO.output(m1,1)
                    GPIO.output(m2,0)
                    GPIO.output(m3,1)
                    GPIO.output(m4,0)
                    i=0
