import face_recognition
import glob
import os
import pickle
import cv2
import numpy as np
import time
from tqdm import tqdm

class Face_wrapper:
    def __init__(self,dir):
        self.dir = dir
        self.known_face_names = []
        self.known_face_encodings = []
        self.process_this_frame = 0
        self._encoding()

    def _encoding(self):
        print('*** encoding face ***')
        image_paths = glob.glob(os.path.join(self.dir,'*.jpg'))

        names = [os.path.basename(path)[:-4] for path in image_paths]

        encoding_file_path = os.path.join(self.dir,'face_encondings.pkl')

        try:
            with open(encoding_file_path,'rb') as f:
                encoding_dict = pickle.load(f)
        except FileNotFoundError as e:
            encoding_dict = {}
        finally:
            for idx,name in tqdm(enumerate(names)):

                if name not in encoding_dict:
                    img = face_recognition.load_image_file(image_paths[idx])
                    try:
                        encoding_dict[name] = face_recognition.face_encodings(img)[0]
                    except IndexError as e:
                        print("*** can not recognize a face from :{}  skipping ***".format(image_paths[idx]))

        with open(encoding_file_path,'wb') as f:
            pickle.dump(encoding_dict,f)
        
        for key,value in encoding_dict.items():
            self.known_face_names.append(key)
            self.known_face_encodings.append(value)

    def recognize(self,image,image_to_draw,resize_ratio=1):
        small_img = cv2.resize(image,(0,0),fx=1,fy=1)
        rgb_small_img = cv2.cvtColor(small_img,cv2.COLOR_BGR2RGB)
        
        if self.process_this_frame%3 == 0:
            self.face_locations = face_recognition.face_locations(rgb_small_img,model='cnn')
            face_encodings = face_recognition.face_encodings(rgb_small_img,self.face_locations)

            self.face_names = []
            for face_encoding in face_encodings:

                matches = face_recognition.compare_faces(self.known_face_encodings,face_encoding)
                name = "Unknown"

                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = self.known_face_names[first_match_index]

                face_distances = face_recognition.face_distance(self.known_face_encodings,face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index].split('_')[0]
                
                self.face_names.append(name)

        self.process_this_frame += 1

        print(self.face_names)

        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size

            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label image a name below the face
            cv2.rectangle(image_to_draw, (left, bottom), (right, bottom+35), (0, 0, 255), cv2.FILLED)
            cv2.putText(image_to_draw, name, (left + 6, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        return  image_to_draw
    
    


        
