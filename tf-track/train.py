import numpy as np
from PIL import Image
import imageio

from tensorflow.keras import Sequential, metrics, optimizers, models
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dropout


w, h = 640, 480

model = Sequential()

model.add(AveragePooling2D(pool_size=(2, 2), input_shape=(h,w,1)))
#model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(h,w,1)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='linear'))
model.compile(learn_rate=0.001, loss='mean_squared_error', metrics=[metrics.MeanAbsoluteError()])


#model.add(MaxPooling2D(pool_size=(2, 2), input_shape=(h,w,1)))
#model.add(Conv2D(32, kernel_size=6, strides=(2,2), activation='relu'))
#model.add(Conv2D(32, kernel_size=6, strides=(2,2), activation='relu'))
#model.add(Conv2D(32, kernel_size=6, strides=(2,2), activation='relu'))
#model.add(Conv2D(8, kernel_size=1, strides=(1,1), activation='relu'))
#model.add(Flatten())
#model.add(Dense(64, activation='relu'))
#model.add(Dense(2, activation='linear'))
#model.compile(learn_rate=0.001, loss='mean_squared_error', metrics=[metrics.MeanAbsoluteError()])

model.summary()

#x_train = []
#y_train = []
#
#x_val = []
#y_val = []
#
#for i in range(0, 63):
#    print(i, "--------------------------------------------------------------------")
#    with open("../data/pupil_vids/data/"+str(i)+".txt") as f:
#        lines = f.readlines()
#        lines = [x.strip() for x in lines]
#        for j, xy in enumerate(lines):
#            # we have too many almost similar frames, so skip some
#            if j%20==0:
#                if j%100==0: print(j)
#                try:
#                    parts = xy.split(" ")
#                    y = [float(parts[1]), float(parts[0])]
#                    impath = "../data/pupil_vids/data/imgs/"+str(i)+"_"+str(j+1).zfill(4)+".png"
#                    img = Image.open(impath).convert('L')
#                    x = np.asarray(img)
#                    x = x[:, :, np.newaxis]
#                    assert not np.any(np.isnan(x))
#                    assert not np.any(np.isnan(y))
#                    if i%15==0:
#                        x_val.append(x)
#                        y_val.append(y)
#                    else:
#                        x_train.append(x)
#                        y_train.append(y)
#                except Exception as e:
#                    print(e)
#                    pass
#
#x_train = np.array(x_train, dtype='float16') / 255.
#y_train = np.array(y_train, dtype='float16')
#x_val = np.array(x_val, dtype='float16') / 255.
#y_val = np.array(y_val, dtype='float16')
#
#print(x_train.shape)
#print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#
#print(x_train[0])
#print(y_train[0])
#
#assert not np.any(np.isnan(x_train))
#assert not np.any(np.isnan(y_train))
#assert not np.any(np.isnan(x_val))
#assert not np.any(np.isnan(y_val))
#
#np.save("../data/pupil_vids/x_train", x_train)
#np.save("../data/pupil_vids/y_train", y_train)
#np.save("../data/pupil_vids/x_val", x_val)
#np.save("../data/pupil_vids/y_val", y_val)

x_val = np.load("../data/pupil_vids/x_val.npy")
y_val = np.load("../data/pupil_vids/y_val.npy")
y_val[:,0] /= 480
y_val[:,1] /= 640

if False:
    x_train = np.load("../data/pupil_vids/x_train.npy")
    y_train = np.load("../data/pupil_vids/y_train.npy")
    y_train[:,0] /= 480
    y_train[:,1] /= 640

    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32)
    model.save('model.h5')

# Recreate the exact same model purely from the file
model = models.load_model('model.h5')
preds = model.predict(x_val)
print("predicted!", preds.shape)
for i, pred in enumerate(preds):
    try:
        x_val[i, int(pred[0]*480), :, 0] = 1.
        x_val[i, :, int(pred[1]*640), 0] = 1.
        imageio.imwrite("preds/"+str(i)+".png", x_val[i])
        print("saved", i)
    except:
        pass



for i in range(0, 1000):
    model.fit_generator(
            train_generator,
            steps_per_epoch=100, # batch_size
            epochs=1,
            validation_data=val_generator,
            validation_steps=100) # batch_size

