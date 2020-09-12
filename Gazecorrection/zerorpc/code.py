import cv2
import numpy as np
import math
import queue
import time
import dlib
import sys
import os

##눈 검출 Left, Right 프레임별 위치

eyeDlibPtL = []
eyeDlibPtR = []

# 저장 프레임 수
frameSave = 20

# 추가
phi = 0
theta = 0.45

# cap = cv2.VideoCapture("testvideo.mp4")    #960,720
w = 960
h = 720

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))

w_r = 15
h_r = 15


plist_prev = []
region_prev = [10, 10, 10, 10]

startflag = 0  # 시작flag(초기값 설정)
warpflag_prev = 0  # 이전프레임 warp 여부 1:wapred 2:original


def pointExtraction(frame, gray, detector, predictor):
    frame_downsampled = frame.copy()
    gray_downsampled = gray.copy()
    frame_downsampled = cv2.resize(frame_downsampled, dsize=(int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)),
                                   interpolation=cv2.INTER_AREA)  # for dlib scale
    gray_downsampled = cv2.resize(gray_downsampled, dsize=(int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5)),
                                  interpolation=cv2.INTER_AREA)  # for dlib scale

    dets = detector(gray_downsampled, 1)  # 업셈플링 횟수

    eyeDlibPtL.clear()
    eyeDlibPtR.clear()
    list_points = []  # detect되는 얼굴 1개라고 가정...*
    for face in dets:
        shape = predictor(frame_downsampled, face)  # 얼굴에서 68개 점 찾기
        for p in shape.parts():
            list_points.append([p.x * 2, p.y * 2])

        list_points = np.array(list_points)  # 리스트를 numpy로 변형
        for idx in range(36, 42):
            eyeDlibPtL.insert(0, list_points[idx])
        for idx in range(42, 48):
            eyeDlibPtR.insert(0, list_points[idx])
    # print(list_points)
    return list_points


def CreateVeg(x_l, y_l, w, h):  # 삭제예정
    Veg = [0] * (640)
    ax = 0
    ay = 0

    for t in range(0, w + 10):
        ax = int(
            (1 - t / (w + 9)) * (1 - t / (w + 9)) * (x_l) +
            (2 * (t / (w + 9)) * (1 - t / (w + 9)) * (x_l + w * 6 / 10)) +
            (t / (w + 9)) * (t / (w + 9)) * (x_l + w))
        ay = int(
            (1 - t / (w + 9)) * (1 - t / (w + 9)) * (int(y_l * 1.05)) +
            (2 * (t / (w + 9)) * (1 - t / (w + 9)) * (y_l - h / 4 + h / 8 + h / 16)) +
            (t / (w + 9)) * (t / (w + 9)) * (y_l + h / 2))

        Veg[ax] = ay

    return Veg


def getTheta(tilt, p_eye, f):  #
    # in case ) tilt = 0 -> theta = p_eye/f
    theta = (
        math.atan(
            (p_eye * math.cos(tilt) - f * math.sin(tilt)) /
            (p_eye * math.sin(tilt) + f * math.cos(tilt))
        )
    )
    return theta


def getRadius(Zn, faceHalf):
    return (Zn * Zn + pow(faceHalf, 2)) / (2 * Zn) + 5


def interpolationVec(h, w, x, y, MaskFrame, frame, ResultFrame, mov_info):
    # Interpolation
    for j in range(0, h - 1):
        for i in range(0, w - 1):
            y_eye = y
            x_eye = x

            if MaskFrame[j + y_eye][i + x_eye][0] == 0 or j == 0:  # 변환정보 flag check j==0(경계: pass)
                continue
            else:
                flag_p = 0  # plus 방향 Vector조사 flag #선언
                flag_m = 0  # minus방향 Vector조사 flag #선언
                y_mov_p = 0  # plus방향 mov #선언
                y_mov_m = 0  # minus방향 mov #선언

                y_mov = 0  # 선언

                tmp = []  # interpolation할 vector 배열 #선언

                for e in range(1, int(h / 2)):  # search range : 1-50 #넉넉하게준것
                    if flag_p == 0:
                        if mov_info[(j + e) * w + i] != 0:  # plus방향 e만큼 이동, mov정보 확인
                            y_mov_p = j + e - mov_info[(j + e) * w + i]
                            tmp.append((y_mov_p, e))
                            flag_p = 1  # 가장 가까운 plus방향 조사 완료, flag 비활성화
                    if flag_m == 0:
                        if i - e >= 0:
                            # if ((i-e)*w+j)>=w*2*h*2:
                            #    print(i,j,e,w,h,len(mov_info))

                            if mov_info[(j - e) * w + i] != 0:  # minus방향 e만큼 이동, mov정보 확인
                                y_mov_m = j - e - mov_info[(j - e) * w + i]
                                tmp.append((y_mov_m, e))
                                flag_m = 1  # 가장 가까운 minus방향 조사 완료, flag 비활성화
                    if flag_p == 1 and flag_m == 1:  # plus, minus방향 모두 조사 완료
                        y_mov = (tmp[0][0] * tmp[1][1] + tmp[1][0] * tmp[0][1]) / (
                                    tmp[0][1] + tmp[1][1])  # bilinear interpolation
                        ResultFrame[j + y_eye][i + x_eye] = frame[j - int(round(y_mov)) + y_eye][i + x_eye]  # 이미지 저장
                        break
    return ResultFrame


def warping(phi, x, y, w, h, cx, cy, frame, ResultFrame):
    ###Matching Z value using Cylinder Model
    MaskFrame = frame.copy()
    f = int(64000 / w)
    # f=500
    w = int(w * 1.2)  # 상수 비율 / 어짜피 곡선으로 자를거면 크게 하던가 or 사용자 처음때 입력.
    x = int(x - w / 5)
    h = int(h * 2.5)
    y = int(y - h / 2)

    mov_info = [0] * (int(w * 2)) * (int(h * 2))  # 변환 이후 정보 저장 배열 , 새로생성

    ZMatrix = np.empty((h, int(w * 1.1)))  # Z값 LUT

    FaceOrigin = (h / 2, w / 2)  # 임시값 #이유 : ZMatrix는 눈영역만 따로만듬.

    faceHalf = w / 2  # 임시값

    Zn = faceHalf  # 임시값

    ###
    Veg = CreateVeg(x, y, w, h)  # 다른 곡선으로바꿀예정
    ###
    theta = getTheta(phi, cy, f)
    # print(theta)
    r = getRadius(Zn, faceHalf)  # Cylinder model 반지름

    for i in range(0, h):  # y = Xb
        for j in range(0, w):  # x = Yb
            x_eye = x
            y_eye = y

            Xb = i
            Yb = j

            w_gaze = -0
            w_eyeheight = 100 / 100

            h_a = int(h / 2)
            w_a = int(w / 2)
            ZMatrix[h_a][w_a] = Zn - r + math.sqrt(pow(r, 2) - pow(w / 2 - FaceOrigin[1], 2))  # Z value 대응
            tmp = int(math.cos(-theta - w_gaze) * h_a + math.sin(-theta - w_gaze) * ZMatrix[h_a][w_a])  # tmp :Y축 회전 X값
            ZMatrix[h_a][w_a] = int(
                math.sin(-theta - w_gaze) * (-1) * h_a + math.cos(-theta - w_gaze) * ZMatrix[h_a][w_a])  # Y축 회전 Z값
            alpha = h_a - tmp  # Y축 회전 이후 이동된 좌표 차이

            ZMatrix[i][j] = Zn - r + math.sqrt(pow(r, 2) - pow(j - FaceOrigin[1], 2))  # Z value 대응

            tmp = int(math.cos(-theta - w_gaze) * i + math.sin(-theta - w_gaze) * ZMatrix[i][j])  # tmp :Y축 회전 X값
            ZMatrix[i][j] = int(
                math.sin(-theta - w_gaze) * (-1) * i + math.cos(-theta - w_gaze) * ZMatrix[i][j])  # Y축 회전 Z값
            v = int((i - h_a) * theta * 1.1)

            Xa_eye = int((w_eyeheight) * int(round(
                tmp * (math.cos(theta) + math.sin(theta) * math.tan(theta)) - ZMatrix[i][j] * math.sin(
                    theta) * math.cos(theta)))) + int(alpha * (
                1.4)) + v  # +int(h*(theta+w_gaze)/2) #X값 변환(Cylinder) #height와 Theta에비례해서 더해줘야함 : alpha: 모든픽셀에대해 차이를 더해주니깐 원래대로돌아옴

            Xa = Xa_eye
            if ((Xa) >= 0) and ((Xa) <= h):  # 변환 범위 지정 : 원래 이미지 크기로 제한

                mov_info[w * Xa + Yb] = Xb  # 변환 좌표정보 입력
                MaskFrame[Xa + y_eye][Yb + x_eye] = 0  # 변환된 좌표 flag 표시.
                ResultFrame[Xa + y_eye][Yb + x_eye] = frame[Xb + y_eye][Yb + x_eye]  # 변환 이미지 출력
                if (Xa + y_eye < Veg[Yb + x_eye]):
                    ResultFrame[Xa + y_eye][Yb + x_eye] = frame[Xa + y_eye][Yb + x_eye]  # 변환 없앰
                else:
                    tmp_diff = ResultFrame[Xa + y_eye][Yb + x_eye] - frame[Xa + y_eye][Yb + x_eye]  # 변환 이미지 출력
                    if math.sqrt(pow(tmp_diff[0], 2) + pow(tmp_diff[1], 2) + pow(tmp_diff[2], 2)) < 40:
                        tmp_1 = ResultFrame[Xa + y_eye][Yb + x_eye]
                        tmp_2 = frame[Xa + y_eye][Yb + x_eye]
                        ResultFrame[Xa + y_eye][Yb + x_eye] = (
                        (int(tmp_1[0]) + int(tmp_2[0])) / 2, (int(tmp_1[1]) + int(tmp_2[1])) / 2,
                        (int(tmp_1[2]) + int(tmp_2[2])) / 2)

    #ResultFrame = interpolationVec(h, w, x, y, MaskFrame, frame, ResultFrame, mov_info)

    return ResultFrame

class videoshow():

    ###############################################################################################################################################
    # I S S U E #



    ###############################################################################################################################################


    ##눈 검출 Left, Right 프레임별 위치

    # eyeDlibPtL =[]
    # eyeDlibPtR =[]
    #
    #
    # #저장 프레임 수
    # frameSave=20
    #
    #
    #
    # #추가
    # phi=0
    # theta = 0.45
    #
    #
    #
    # #cap = cv2.VideoCapture("testvideo.mp4")    #960,720
    # w = 960
    # h = 720
    #
    # RIGHT_EYE = list(range(36, 42))
    # LEFT_EYE = list(range(42, 48))
    #
    #
    # w_r=15
    # h_r=15
    #
    # mvinfo_prev=[]
    # plist_prev=[]
    # region_prev=[10,10,10,10]
    #
    # startflag=0#시작flag(초기값 설정)
    # warpflag_prev=0 #이전프레임 warp 여부 1:wapred 2:original

    ############################################################################################################
    # def pointExtraction(frame,gray,detector,predictor):
    #     frame_downsampled = frame.copy()
    #     gray_downsampled = gray.copy()
    #     frame_downsampled = cv2.resize(frame_downsampled,dsize=(int(gray.shape[1]*0.5), int(gray.shape[0]*0.5)),interpolation=cv2.INTER_AREA) #for dlib scale
    #     gray_downsampled = cv2.resize(gray_downsampled,dsize=(int(gray.shape[1]*0.5), int(gray.shape[0]*0.5)),interpolation=cv2.INTER_AREA) #for dlib scale
    #
    #     dets = detector(gray_downsampled, 1) # 업셈플링 횟수
    #
    #     eyeDlibPtL.clear()
    #     eyeDlibPtR.clear()
    #     list_points = []# detect되는 얼굴 1개라고 가정...*
    #     for face in dets:
    #         shape = predictor(frame_downsampled, face) #얼굴에서 68개 점 찾기
    #         for p in shape.parts():
    #             list_points.append([p.x*2, p.y*2])
    #
    #         list_points = np.array(list_points) # 리스트를 numpy로 변형
    #         for idx in range(36,42):
    #             eyeDlibPtL.insert(0,list_points[idx])
    #         for idx in range(42, 48):
    #             eyeDlibPtR.insert(0,list_points[idx])
    #     #print(list_points)
    #     return list_points
    #
    # def CreateVeg(x_l,y_l,w,h): #삭제예정
    #     Veg = [0]*(640)
    #     ax=0
    #     ay=0
    #
    #     for t in range(0,w+10):
    #         ax= int(
    #             (1-t/(w+9)) * (1-t/(w+9)) * (x_l) +
    #               (2 * (t/(w+9)) * (1-t/(w+9)) * (x_l+w*6/10)) +
    #               (t/(w+9)) * (t/(w+9)) * (x_l+w))
    #         ay=int(
    #             (1-t/(w+9)) * (1-t/(w+9)) * (int(y_l*1.05)) +
    #               (2 * (t/(w+9)) * (1-t/(w+9)) * (y_l-h/4+h/8+h/16)) +
    #               (t/(w+9)) * (t/(w+9)) * (y_l+h/2))
    #
    #         Veg[ax]=ay
    #
    #
    #     return Veg
    #
    # def getTheta(tilt, p_eye,f): #
    #     # in case ) tilt = 0 -> theta = p_eye/f
    #     theta = (
    #         math.atan(
    #             (p_eye*math.cos(tilt) - f*math.sin(tilt))/
    #              (p_eye*math.sin(tilt) + f*math.cos(tilt))
    #             )
    #              )
    #     return theta
    #
    # def getRadius(Zn,faceHalf):
    #     return (Zn * Zn + pow(faceHalf, 2)) / (2* Zn)+5
    #
    # def interpolationVec(h,w,x,y,MaskFrame,frame,ResultFrame,mov_info):
    #     #Interpolation
    #     for j in range(0,h-1):
    #         for i in range(0,w-1):
    #             y_eye=y
    #             x_eye=x
    #
    #             if MaskFrame[j+y_eye][i+x_eye][0] == 0 or j==0: #변환정보 flag check j==0(경계: pass)
    #                 continue
    #             else:
    #                 flag_p=0 #plus 방향 Vector조사 flag #선언
    #                 flag_m=0 #minus방향 Vector조사 flag #선언
    #                 y_mov_p=0 #plus방향 mov #선언
    #                 y_mov_m=0 #minus방향 mov #선언
    #
    #                 y_mov=0 #선언
    #
    #                 tmp=[] #interpolation할 vector 배열 #선언
    #
    #                 for e in range(1,int(h/2)):#search range : 1-50 #넉넉하게준것
    #                     if flag_p==0:
    #                         if mov_info[(j+e)*w+i] != 0: #plus방향 e만큼 이동, mov정보 확인
    #                             y_mov_p= j+e - mov_info[(j+e)*w+i]
    #                             tmp.append((y_mov_p,e))
    #                             flag_p=1#가장 가까운 plus방향 조사 완료, flag 비활성화
    #                     if flag_m==0:
    #                         if i-e>=0:
    #                             #if ((i-e)*w+j)>=w*2*h*2:
    #                             #    print(i,j,e,w,h,len(mov_info))
    #
    #                             if mov_info[(j-e)*w+i]!=0: #minus방향 e만큼 이동, mov정보 확인
    #                                 y_mov_m= j-e - mov_info[(j-e)*w+i]
    #                                 tmp.append((y_mov_m,e))
    #                                 flag_m=1#가장 가까운 minus방향 조사 완료, flag 비활성화
    #                     if flag_p==1 and flag_m==1:#plus, minus방향 모두 조사 완료
    #                         y_mov=(tmp[0][0]*tmp[1][1]+tmp[1][0]*tmp[0][1])/(tmp[0][1]+tmp[1][1]) #bilinear interpolation
    #                         ResultFrame[j+y_eye][i+x_eye] = frame[j-int(round(y_mov))+y_eye][i+x_eye] #이미지 저장
    #                         break
    #     return ResultFrame
    #
    # def warping(phi,x,y,w,h,cx,cy,frame,ResultFrame):
    #
    #     ###Matching Z value using Cylinder Model
    #     f=int(64000/w)
    #     #f=500
    #     w=int(w*1.2) # 상수 비율 / 어짜피 곡선으로 자를거면 크게 하던가 or 사용자 처음때 입력.
    #     x=int(x-w/5)
    #     h=int(h*2.5)
    #     y=int(y-h/2)
    #
    #     mov_info=[0]*(int(w*2))*(int(h*2)) #변환 이후 정보 저장 배열 , 새로생성
    #
    #     ZMatrix = np.empty((h, int(w*1.1))) #Z값 LUT
    #
    #     FaceOrigin = ( h/2, w/2 ) #임시값 #이유 : ZMatrix는 눈영역만 따로만듬.
    #
    #     faceHalf = w / 2 # 임시값
    #
    #     Zn = faceHalf # 임시값
    #
    #     ###
    #     Veg = CreateVeg(x,y,w,h) # 다른 곡선으로바꿀예정
    #     ###
    #     theta = getTheta(phi,cy,f)
    #     #print(theta)
    #     r=getRadius(Zn,faceHalf) # Cylinder model 반지름
    #
    #     for i in range(0,h): #y = Xb
    #         for j in range(0, w): #x = Yb
    #             x_eye=x
    #             y_eye=y
    #
    #             Xb = i
    #             Yb = j
    #
    #             w_gaze=-0
    #             w_eyeheight=100/100
    #
    #             h_a=int(h/2)
    #             w_a=int(w/2)
    #             ZMatrix[h_a][w_a] = Zn - r + math.sqrt(pow(r, 2) - pow(w/2 - FaceOrigin[1], 2)) # Z value 대응
    #             tmp = int(math.cos(-theta-w_gaze)*h_a + math.sin(-theta-w_gaze)*ZMatrix[h_a][w_a]) #tmp :Y축 회전 X값
    #             ZMatrix[h_a][w_a]= int(math.sin(-theta-w_gaze)*(-1)*h_a+ math.cos(-theta-w_gaze)*ZMatrix[h_a][w_a]) #Y축 회전 Z값
    #             alpha= h_a - tmp # Y축 회전 이후 이동된 좌표 차이
    #
    #
    #             ZMatrix[i][j] = Zn - r + math.sqrt(pow(r, 2) - pow(j - FaceOrigin[1], 2)) # Z value 대응
    #
    #             tmp = int(math.cos(-theta-w_gaze)*i + math.sin(-theta-w_gaze)*ZMatrix[i][j]) #tmp :Y축 회전 X값
    #             ZMatrix[i][j] = int(math.sin(-theta-w_gaze)*(-1)*i+ math.cos(-theta-w_gaze)*ZMatrix[i][j]) #Y축 회전 Z값
    #             v=int((i-h_a)*theta*1.1)
    #
    #             Xa_eye = int((w_eyeheight)*int(round( tmp * (math.cos(theta) + math.sin(theta) * math.tan(theta)) - ZMatrix[i][j] * math.sin(theta) * math.cos(theta) ) )) + int(alpha*(1.4))+v#+int(h*(theta+w_gaze)/2) #X값 변환(Cylinder) #height와 Theta에비례해서 더해줘야함 : alpha: 모든픽셀에대해 차이를 더해주니깐 원래대로돌아옴
    #
    #             Xa=Xa_eye
    #             if ((Xa) >= 0) and ((Xa) <= h): #변환 범위 지정 : 원래 이미지 크기로 제한
    #
    #                 mov_info[w*Xa+Yb]=Xb #변환 좌표정보 입력
    #                 MaskFrame[Xa+y_eye][Yb+x_eye]=0  #변환된 좌표 flag 표시.
    #                 ResultFrame[Xa+y_eye][Yb+x_eye] = frame[Xb+y_eye][Yb+x_eye] #변환 이미지 출력
    #                 if(Xa+y_eye < Veg[Yb+x_eye]):
    #                         ResultFrame[Xa+y_eye][Yb+x_eye] = frame[Xa+y_eye][Yb+x_eye] #변환 없앰
    #                 else:
    #                     tmp_diff= ResultFrame[Xa+y_eye][Yb+x_eye] - frame[Xa+y_eye][Yb+x_eye] #변환 이미지 출력
    #                     if math.sqrt( pow(tmp_diff[0],2)+pow(tmp_diff[1],2)+pow(tmp_diff[2],2) ) < 40:
    #                         tmp_1=ResultFrame[Xa+y_eye][Yb+x_eye]
    #                         tmp_2=frame[Xa+y_eye][Yb+x_eye]
    #                         ResultFrame[Xa+y_eye][Yb+x_eye] = ((int(tmp_1[0])+int(tmp_2[0]))/2,(int(tmp_1[1])+int(tmp_2[1]))/2,(int(tmp_1[2])+int(tmp_2[2]))/2)
    #
    #
    #     ResultFrame = interpolationVec(h,w,x,y,MaskFrame,frame,ResultFrame,mov_info)
    #
    #     return ResultFrame

    ################################################################################
    def viewing(self):
        cap = cv2.VideoCapture("testvideo.mp4")

        ##전역이어야하는데 전역으로 인식안되는변수들
        region_prev = [10, 10, 10, 10]
        startflag = 0
        plist_prev = []

        ##
        print("first code test")
        ret, frame_prev=cap.read()
        while(not ret):
            ret, frame_prev = cap.read()
        gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

        prevTime = 0

        Result_prev=frame_prev.copy()

        ##################tmp for frame rate up conversion

        pad=50
        ################################################

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 학습모델 로드

        # range는 끝값이 포함안/특징마다 번호 부여
        RIGHT_EYE = list(range(36, 42))
        LEFT_EYE = list(range(42, 48))

        index = LEFT_EYE + RIGHT_EYE



        ########## Cam Loop#####################################################################################################################################################################################
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                gray = cv2.blur(gray, (3, 3), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
                gray = cv2.equalizeHist(gray)
                ##faces = cv2.CascadeClassifier('C:/xml/haarcascade_frontalface_default.xml')
                #eyes = cv2.CascadeClassifier('C:/xml/haarcascade_eye.xml')


                # 1. Eye Detecting
                list_points = pointExtraction(frame,gray,detector,predictor)

                ###################################################################################

                ResultFrame=frame.copy()
                TrackingFrame = frame.copy() # Warping 결과 Frame
                #MaskFrame=frame.copy() # Warping 값 flag 배열(for interpolation)때문에 잠깐 뺐었는데 다시 넣을것

                #detected_f = faces.detectMultiScale(gray, 1.3, 5) #Face(detected)
                #detected = eyes.detectMultiScale(gray, 1.3, 5) #Eyes(detected)
                warpflag=1 #warp초기값
                if len(list_points)<68:
                    #wapredflag==0으로
                    #print("warpedflag=0 or can't find eye or etc...")
                    #print(list_points)
                    warpflag=0
                else:
                    x=list_points[36][0]
                    y=int((list_points[37][1]+list_points[38][1])/2)
                    w=int((list_points[39][0]-x)*1.5)
                    h=int((list_points[41][1]-y)*1.5)

                    #right eye
                    x_r=list_points[42][0]
                    y_r=int((list_points[43][1]+list_points[44][1])/2)
                    w_r=int((list_points[45][0]-x_r)*1.5)
                    h_r=int((list_points[47][1]-y_r)*1.5)


                    #픽셀값저장
                    ##############startflag에서는 안하고 검출이후 안정화에서 쓰일예정이니 초기값은 일단 임시로생각해도될듯
                    diffsum=999
                    x_p,y_p,w_p,h_p=region_prev
                    for i in range(y_p,y_p+h_p):
                        for j in range(x_p,x_p+w_p):
                            diffsum= diffsum + abs(int(gray[i][j])-int(gray_prev[i][j]))
                    diffsum = diffsum/(w_p*h_p)
                    if diffsum<8 and len(plist_prev) != 0:
                        print("not move")
                        list_points = plist_prev


                    ###문제점 : 정확한 영역 추출이 불가능함.. 일단 와핑영역, 기타 추정치에대해 사용은 가능하지만 eye shape추정하는게 힘들어짐
                    cnt=0#임시로일단 표시하려고
                    for i in list_points:
                        #cv2.circle(frame, (i[0],i[1]), 2, (0, 255, 0), -1)
                        #cv2.putText(frame, str(cnt),(i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
                        cnt=cnt+1


                    #cv2.imshow("gray",gray)
                    #cv2.imshow("gray2",gray_prev)
                    region_prev=[x,y,w,h]


                # 2. Gaze Estimation
                    #gazeVec = ExtractGazeVector() #작성 예정

                if startflag == 0:
                    print("start")

                    startflag = 1
                else:
                    if warpflag==1 :

                        #warpflag_prev 판단

                        cx,cy=tuple(np.average(eyeDlibPtL,0))
                        cx_r,cy_r=tuple(np.average(eyeDlibPtR,0))
                        #cv2.circle(frame,(int(cx),int(cy)),2,(0,0,255),-1)
                        #cv2.circle(frame,(int(cx_r),int(cy_r)),2,(0,0,255),-1)

                        ############################################

                        #left eye
                        x=list_points[36][0]
                        y=int((list_points[37][1]+list_points[38][1])/2)
                        w=int((list_points[39][0]-x)*1.5)
                        h=int((list_points[41][1]-y)*1.5)

                        #right eye
                        x_r=list_points[42][0]
                        y_r=int((list_points[43][1]+list_points[44][1])/2)
                        w_r=int((list_points[45][0]-x_r)*1.5)
                        h_r=int((list_points[47][1]-y_r)*1.5)

                        ##########################################



                        #3. Warping
                        ResultFrame = warping(phi,x,y,w,h,cx,cy,frame,ResultFrame)


                        ##4. Warping
                        ResultFrame=warping(phi,x_r,y_r,w_r,h_r,cx_r,cy_r,frame,ResultFrame)

                    else : # warpflag 0일 때
                        #warpflag_prev 판단
                        print("not warp")


                plist_prev = list_points
                gray_prev = gray.copy()
                frame_prev=frame.copy() ##이전프레임 frame rate up을 위한

                Result_prev = ResultFrame.copy()

                ResultFrame = cv2.medianBlur(ResultFrame, 3)
                frame = cv2.medianBlur(frame, 3)
                curTime = time.time()
                sec = curTime - prevTime
                prevTime = curTime

                fps = 1 / sec
                str_fps = "FPS : %0.1f" % fps
                cv2.putText(frame, str_fps, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                # cv2.imshow("prev",Result_prev)
                # cv2.imshow('TrackingFrame',TrackingFrame)
                cv2.imshow('Frame',frame)
                cv2.imshow('ResultFrame',ResultFrame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        cap.release()
        cv2.destroyAllWindows()

    ####################################################################################################################################################################################################


import zerorpc

s= zerorpc.Server(videoshow())
s.bind("tcp://0.0.0.0:4242")
s.run()
