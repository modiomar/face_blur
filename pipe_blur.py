import cv2 as cv
import mediapipe as mp
import os
import numpy as np

#a list of the face mesh points, making up the face outline polygon 
outer_points = [10, 338, 297, 332, 284, 447, 288, 365, 378, 400, 152, 148, 176, 149, 150, 136, 172, 215, 177, 137,
                162, 21, 54, 103, 67, 109]

#collecting the names of the photos in the 'faces' directory, then iterating over them
image_names = os.listdir('faces')

blurred_image_names = os.listdir('blurred')

if not image_names:
    raise ValueError('There are no files in the folder faces')

for image_name in image_names:

    #load and convert image to RGB, create a blurred copy
    img = cv.imread(os.path.join('faces', image_name))
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    blur = cv.blur(img.copy(), (151, 151))

    #initializing face mesh detection objects
    mp_mesh = mp.solutions.face_mesh
    mesh = mp_mesh.FaceMesh(max_num_faces=20, min_detection_confidence=0.5)

    #finding all face meshes in the RGB image
    results = mesh.process(rgb_img)

    h, w, c = rgb_img.shape

    #the list to contain the lists of faces, containing landmark x-y coordinates in them
    faces = []
    #if faces exist, iterate over them
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            face = []
            #iterating over landmarks in each individual face
            face_lms_ls = face_lms.landmark
            for i in outer_points:
                #appending the de-normalized landmark x-y coordinates list to the face list
                face.append([int(w*face_lms_ls[i].x), int(h*face_lms_ls[i].y)])
            #adding the face list to the list of faces
            faces.append(face)

    #creating the mask, then drawing the filled in face outline polygons   
    mask = np.zeros_like(img)         
    for face in faces:
        arr = np.int32([face])
        cv.fillPoly(mask, pts=arr, color=(255, 255, 255))
        #applying the mask
        img_blurred = np.where(mask==np.array([255, 255, 255]), blur, img)

    #if this image hasn't been blurred before, write it to the 'blurred' folder
    if "blurred_" + image_name in blurred_image_names:
        continue
    else:
        cv.imwrite(os.path.join('blurred', "blurred_" + image_name), img_blurred)