import numpy as np
from PIL import Image

npzfiles = np.load("../data/eye_tracker_train_and_val.npz")
print(npzfiles.files)
# ['train_y', 'train_eye_right', 'train_face', 'train_eye_left', 'val_face', 'val_eye_right', 'train_face_mask', 'val_y', 'val_face_mask', 'val_eye_left']

t_y = npzfiles['train_y']
t_er = npzfiles['train_eye_right']

print(t_y.shape)
print(t_er.shape)

for i, y in enumerate(t_y):
    if i < 100:
        print(y)

for i, er in enumerate(t_er):
    if i < 100:
        im = Image.fromarray(er)
        im.save("test/"+str(i)+".jpeg")
