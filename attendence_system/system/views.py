from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import cv2
import numpy as np

# Create your views here.

def system(request):
    template = loader.get_template('base.html')
    return HttpResponse(template.render())

def recognition(request):
    template = loader.get_template('recognition.html')
    video_capture = cv2.VideoCapture(0) 
    if request.method == 'POST':  
        
        while True:  
            ret, frame = video_capture.read()  
            rgb_frame = frame[:, :, ::-1]  

            # Nhận diện khuôn mặt  
            # face_locations = face_recognition.face_locations(rgb_frame)  
            # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  

            # students = Student.objects.all()  
            # for student in students:  

            #     known_face_encodings = np.frombuffer(student.face_encoding)  

            #     for face_encoding in face_encodings:  
            #         matches = face_recognition.compare_faces([known_face_encodings], face_encoding)  
            #         if True in matches:  
            #             # Thực hiện điểm danh  
            #             print(f'{student.name} đã điểm danh.')  
            
            # Hiển thị khung hình cho video  
            cv2.imshow('Video', frame)  
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break  

    video_capture.release()  
    cv2.destroyAllWindows() 
    return render(request, 'recognition.html')

def get_students(request):

    return render(request, 'get_list.html')