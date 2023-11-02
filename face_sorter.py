#!/usr/bin/python3

from zlib import crc32
import numpy as np
import face_recognition as fr
from PIL import Image, ImageOps, ImageDraw
import os
import shutil
import signal
import time


class FaceSorter:

    image_file_types = ('.jpg', '.JPG', '.jpeg')
    image_files = []
    
    # List of known face_encoding
    known_face_encodings = []

    # Corresponding table between face_encoding (crc32 encoded) & face_id
    known_face_id = {}


    def __init__(self):
        self.source_directory = os.path.abspath('input')
        self.target_directory = os.path.abspath('output')
        self.debug_directory = os.path.abspath('debug')

    def process_image(self, file):
        print(f'Started processing {file}...')
        try:
            image = fr.load_image_file(file)
            face_locations = fr.face_locations(image, number_of_times_to_upsample=0, model='cnn')
            face_encodings = fr.face_encodings(image, face_locations, 50, "large")
            print(f'Processed {file}...')
            return tuple(zip(face_locations, face_encodings))
        except Exception as err:
            print(f'ERROR: {err}')

    def detect_face_id(self, face_encoding):
        matches = fr.compare_faces(self.known_face_encodings, face_encoding, 0.45)
        # If any known face was detected, let's resolve it
        if True in matches:
            face_distances = fr.face_distance(self.known_face_encodings, face_encoding)
            match_index = np.argmin(face_distances)
            print(f'Found best match: {match_index}')
            matched_face_encoding = self.known_face_encodings[match_index]
            face_id = self.known_face_id[crc32(matched_face_encoding)]
            print(f'Found known face: {face_id}')
        # Else create a new face in our database
        else:
            face_id = hex(crc32(face_encoding))
            if not os.path.exists(os.path.join(self.target_directory, face_id)):
                os.mkdir(os.path.join(self.target_directory, face_id))
            print(f'Created new face: {face_id}')

        return face_id

    def draw_label_on_image(self, image, name, face_location):
        (top, right, bottom, left) = face_location

        # Draw a box around the face using the Pillow module
        image.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        (text_left, text_top, text_right, text_bottom) = image.textbbox((0, 0), name)
        image.rectangle(((left, bottom - (text_bottom-text_top) - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        image.text((left + 6, bottom - (text_bottom-text_top) - 5), name, fill=(255, 255, 255, 255))

    def sort(self):
        for file in os.listdir(self.source_directory):
            if file.endswith(self.image_file_types):
                full_file = os.path.join(self.source_directory, file)
                print(f'Running {full_file}...')
                data = self.process_image(full_file)

                # If no face are found, let's move on
                if len(data) == 0:
                    print('Found no face')
                    if not os.path.exists(os.path.join(self.target_directory, 'no_face_found')):
                        os.mkdir(os.path.join(self.target_directory, 'no_face_found'))
                    shutil.copyfile(full_file, os.path.join(self.target_directory, 'no_face_found', os.path.basename(full_file)))
                else:
                    pil_image = Image.open(full_file)
                    # Create a Pillow ImageDraw Draw instance to draw with
                    image = ImageDraw.Draw(pil_image)
                    for (face_location, face_encoding) in data:
                        # Detect face id from already detected faces
                        face_id = self.detect_face_id(face_encoding)
                        
                        # Save the new face to our faces database
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_id[crc32(face_encoding)] = face_id

                        # DEBUG: draw face with their detected name
                        self.draw_label_on_image(image, face_id, face_location)
                        # Copy the picture to the coresponding face folder
                        # shutil.copyfile(full_file, os.path.join(self.target_directory, face_id, os.path.basename(full_file)))
                    
                    pil_image = ImageOps.exif_transpose(pil_image)
                    pil_image.save(os.path.join(self.debug_directory, os.path.basename(full_file)))
                    del image


def exit_gracefully(_signal, _frame):
    print('\rGoodbye!')
    exit()


if __name__ == '__main__':
    start = time.perf_counter()

    # Catch CTRL+C and exit in a nice way
    signal.signal(signal.SIGINT, exit_gracefully)

    face_off = FaceSorter()
    face_off.sort()

    finish = time.perf_counter()
    print(f'Finished in {round(finish - start, 2)} second(s)')
