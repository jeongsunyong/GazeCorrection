import cv2
import numpy as np
import math
import queue
import time
import dlib
import zerorpc
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
###############################################################################################################################################
# I S S U E #



###############################################################################################################################################
mod = SourceModule("""
            #include<math.h>
            #include<stdlib.h>

            __device__ void computeCurvePt(float* curveP, int pt, int resultPt[2])
            {
                //int* resultPt = (int*)malloc(2 * sizeof(int));
                //int resultPt[2];
                resultPt[0] = pt;
                resultPt[1] = int(curveP[0] * pt * pt + curveP[1] * pt + curveP[2]);

                //return resultPt;
            }

            // 근의 공식 이용해서 y좌표 주어졌을 때 곡선과 만나는 x좌표 찾기 
            //__device__ void computeCurvePx(float* UpperCurve, float* LowerCurve, int py, int* curvePx_prev)
            __device__ void computeCurvePx(float* UpperCurve, float* LowerCurve, int py, int curvePx_prev[2], int Curve[2])
            {
                if((UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py) < 0) || (LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py)  < 0))
                {
                    Curve[0] = curvePx_prev[0];//
                    Curve[1] = curvePx_prev[1];//
                    //return circlePx_prev;
                }
                else
                {
                    int x1_Up = int((-UpperCurve[1] - sqrtf(UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py))) / (2 * UpperCurve[0]));
                    int x2_Up = int((-UpperCurve[1] + sqrtf(UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py))) / (2 * UpperCurve[0]));
                    int x1_Low = int((-LowerCurve[1] + sqrtf(LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py))) / (2 * LowerCurve[0]));
                    int x2_Low = int((-LowerCurve[1] - sqrtf(LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py))) / (2 * LowerCurve[0]));
                    
                    //int* result = (int*)malloc(2 * sizeof(int));
                    int result[2];
                    result[0] = (x1_Up < x1_Low ? x1_Low : x1_Up);
                    result[1] = (x2_Low > x2_Up ? x2_Up : x2_Low);

                    Curve[0] = result[0];//
                    Curve[1] = result[1];//
                    
                    //return result;
                }
            }

            // y좌표가 주어졌을 때 눈동자와 만나는 점 찾기 (O: 눈동자 원점, R: 반지름, py: 주어진 y좌표)
            //__device__ void computeCirclePx(int O[2], int R, int py, int* circlePx_prev)
            __device__ void computeCirclePx(int O[2], int R, int py, int circlePx_prev[2], int Circle[2])
            {
                if(R * R - (py - O[0]) * (py - O[0]) < 0)
                {
                    Circle[0] = circlePx_prev[0];//
                    Circle[1] = circlePx_prev[1];//
                    //return circlePx_prev;
                }
                else
                {
                    //int* result = (int*)malloc(2 * sizeof(int));
                    int result[2];
                    result[0] = int(sqrtf(R * R - (py - O[0]) * (py - O[0])) + O[1]); // 큰 x
                    result[1] = int(-sqrtf(R * R - (py - O[0]) * (py - O[0])) + O[1]); // 작은 x
                    Circle[0] = result[0];//
                    Circle[1] = result[1];//
                    //return result;
                }
            }

            __device__ bool IsLowerComparison(float* curveP, int pt[2])
            {
                if(pt[1] < (curveP[0] * pt[0] * pt[0] + curveP[1] * pt[0] + curveP[2]))
                    return true;
                else
                    return false;
            }

            __device__ bool IsUpperComparison(float* curveP, int pt[2])
            {
                if(pt[1] > (curveP[0] * pt[0] * pt[0] + curveP[1] * pt[0] + curveP[2]))
                    return true;
                else
                    return false;
            }


            __global__ void pupilCheckL(int* PupilLocationLx_gpu, int* PupilLocationLy_gpu, int* frame_gpu, float* upperCurve, float* lowerCurve, int xmin, int ymin, int pupilcols, int cols, int tempw, int temph)
            {
                if(threadIdx.x + blockDim.x * blockIdx.x < tempw && threadIdx.y + blockDim.y * blockIdx.y < temph)
                {
                    int i = xmin + threadIdx.x + blockDim.x * blockIdx.x;
                    int j = ymin + threadIdx.y + blockDim.y * blockIdx.y;
                    int k = threadIdx.x + blockDim.x * blockIdx.x;
                    int l = threadIdx.y + blockDim.y * blockIdx.y;
                    int Pt[2] = {i, j};

                    if (IsLowerComparison(lowerCurve, Pt) && IsUpperComparison(upperCurve, Pt))
                    {
                        if(frame_gpu[(j * cols + i) * 3] < 80 && frame_gpu[(j * cols + i) * 3 + 1] < 80 && frame_gpu[(j * cols + i) * 3 + 2] < 80)
                        {
                            PupilLocationLx_gpu[l * pupilcols + k] = i;
                            PupilLocationLy_gpu[l * pupilcols + k] = j;
                            //frame_gpu[(j * cols + i) * 3] = 255;
                            //frame_gpu[(j * cols + i) * 3 + 1] = 255;
                            //frame_gpu[(j * cols + i) * 3 + 2] = 255;
                        }
                    }
                }
            }

            __global__ void pupilCheckR(int* PupilLocationRx_gpu, int* PupilLocationRy_gpu, int* frame_gpu, float* upperCurve, float* lowerCurve, int xmin, int ymin, int pupilcols, int cols, int tempw, int temph)
            {
                if(threadIdx.x + blockDim.x * blockIdx.x < tempw && threadIdx.y + blockDim.y * blockIdx.y < temph)
                {
                    int i = xmin + threadIdx.x + blockDim.x * blockIdx.x;
                    int j = ymin + threadIdx.y + blockDim.y * blockIdx.y;
                    int k = threadIdx.x + blockDim.x * blockIdx.x;
                    int l = threadIdx.y + blockDim.y * blockIdx.y;

                    int Pt[2] = {i, j};
                    if (IsLowerComparison(lowerCurve, Pt) && IsUpperComparison(upperCurve, Pt))
                    {                    
                        if(frame_gpu[(j * cols + i) * 3] < 80 && frame_gpu[(j * cols + i) * 3 + 1] < 80 && frame_gpu[(j * cols + i) * 3 + 2] < 80)
                        {
                            PupilLocationRx_gpu[l * pupilcols + k] = i;
                            PupilLocationRy_gpu[l * pupilcols + k] = j;
                            //frame_gpu[(j * cols + i) * 3] = 255;
                            //frame_gpu[(j * cols + i) * 3 + 1] = 255;
                            //frame_gpu[(j * cols + i) * 3 + 2] = 255;
                        }
                    }
                }
            }

            __global__ void warping(int* frame_gpu, float* ZMatrix_gpu, int* mov_info_gpu, int* Veg_gpu, int* resultFrame_gpu, int* MaskFrame_gpu, float FaceOrigin_gpu, int x_eye, int y_eye, float Zn, float r, float theta, int h, int w, int w_tmp, int cols, int rows, float w_gaze)
            {
                int j = threadIdx.x + blockDim.x * blockIdx.x;
                int i = threadIdx.y + blockDim.y * blockIdx.y;
                if((i < h) && (j < w) && (i >= 0) && (j >= 0))
                {
                //float w_gaze = 0.3; // 밖으로 빼도 됨
                float w_eyeheight = 100/float(100); // 밖으로 빼도 됨
                int h_a = int(h/2); // 밖으로 빼도 됨
                int w_a = int(w/2); // 밖으로 빼도 됨
                int tmp_diff[3];
                int tmp_1[3];
                int tmp_2[3];
                theta = 0.5;
                ZMatrix_gpu[h_a * w_tmp + w_a] = Zn - r + sqrtf(r * r - (w/2 - FaceOrigin_gpu) * (w/2 - FaceOrigin_gpu)); // 밖으로 빼도 됨
                int tmp = int(cos(-theta-w_gaze)*h_a + sin(-theta-w_gaze)*ZMatrix_gpu[h_a * w_tmp + w_a]); // 밖으로 빼도 됨

                ZMatrix_gpu[h_a * w_tmp + w_a] = int(sin(-theta-w_gaze)*(-1)*h_a+ cos(-theta-w_gaze)*ZMatrix_gpu[h_a * w_tmp + w_a]); // 밖으로 빼도 됨
                int alpha = h_a - tmp; // 밖으로 빼도 됨
                ZMatrix_gpu[i * w_tmp + j] = Zn - r + sqrtf(r * r - (j - FaceOrigin_gpu) * (j - FaceOrigin_gpu));
                tmp = int(cos(-theta-w_gaze)*i + sin(-theta-w_gaze)*ZMatrix_gpu[i * w_tmp + j]);
                ZMatrix_gpu[i * w_tmp + j] = int(sin(-theta-w_gaze)*(-1)*i+ cos(-theta-w_gaze)*ZMatrix_gpu[i * w_tmp + j]);

                int v = int((i-h_a)*theta*1.1);

                int Xa_eye = int((w_eyeheight)*int(round( tmp * (cos(theta) + sin(theta) * tan(theta)) - ZMatrix_gpu[i * w_tmp + j] * sin(theta) * cos(theta) ) )) + int(alpha*(1.4)) + v;
                int Xa = Xa_eye;

                if((Xa > -1) && (Xa < h + 1))
                {
                    mov_info_gpu[w * Xa + j] = i;
                    MaskFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3] = 0;
                    MaskFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1] = 0;
                    MaskFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2] = 0;
                    resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3] = frame_gpu[((i + y_eye) * cols + j + x_eye) * 3];
                    resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1] = frame_gpu[((i + y_eye) * cols + j + x_eye) * 3 + 1];
                    resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2] = frame_gpu[((i + y_eye) * cols + j + x_eye) * 3 + 2];
                    
                    if(Xa+y_eye < Veg_gpu[j+x_eye])
                    {
                        resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3] = frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3];
                        resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1] = frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3 + 1];
                        resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2] = frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3 + 2];
                    }
                    else
                    {
                        tmp_diff[0] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3] - frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3];
                        tmp_diff[1] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1] - frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3 + 1];
                        tmp_diff[2] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2] - frame_gpu[((Xa+y_eye) * cols + j+x_eye) * 3 + 2];

                        if(sqrtf(tmp_diff[0] * tmp_diff[0] + tmp_diff[1] * tmp_diff[1] + tmp_diff[2] * tmp_diff[2] ) < 40)
                        {
                            tmp_1[0] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3];
                            tmp_1[1] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1];
                            tmp_1[2] = resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2];

                            tmp_2[0] = frame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3];
                            tmp_2[1] = frame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1];
                            tmp_2[2] = frame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2];

                            resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3] = int((tmp_1[0]+tmp_2[0])/2);
                            resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 1] = int((tmp_1[1]+tmp_2[1])/2);
                            resultFrame_gpu[((Xa + y_eye) * cols + j + x_eye) * 3 + 2] = int((tmp_1[2]+tmp_2[2])/2);
                            
                        }
                    }
                }
                }
            }

            __global__ void interpolation(int* frame_gpu, int* mov_info_gpu, int* resultFrame_gpu, int* MaskFrame_gpu, int x, int y, int h, int w, int cols, int rows)
            {
                int i = threadIdx.x + blockDim.x * blockIdx.x;
                int j = threadIdx.y + blockDim.y * blockIdx.y;

                if((i < w - 1) && (j < h - 1) && (i >= 0) && (j >= 0))
                {
                if(MaskFrame_gpu[((j+y) * cols + i+x) * 3] != 0 && j!=0)
                {
                    int flag_p=0; //plus 방향 Vector조사 flag #선언
                    int flag_m=0; //minus방향 Vector조사 flag #선언
                    int y_mov_p=0; //plus방향 mov #선언
                    int y_mov_m=0; //minus방향 mov #선언
                    int y_mov=0;
                    int tmp[4];
                    int tmpCheck = 0;

                    int e;
                    for(e = 1; e < int(h/2); e++)
                    {
                        if(flag_p==0)
                        {
                            if(mov_info_gpu[(j+e)*w+i] != 0)
                            {
                                y_mov_p = j+e - mov_info_gpu[(j+e)*w+i];
                                tmp[tmpCheck] = y_mov_p;
                                tmp[tmpCheck + 1] = e;
                                tmpCheck += 2;
                                flag_p = 1;
                            }
                        }
                        if(flag_m==0)
                        {
                            if(j-e>=0)
                            {
                                if(mov_info_gpu[(j-e)*w+i]!=0)
                                {
                                    y_mov_m = j-e - mov_info_gpu[(j-e)*w+i] ;
                                    tmp[tmpCheck] = y_mov_m;
                                    tmp[tmpCheck + 1] = e;
                                    tmpCheck += 2;
                                    flag_m = 1;
                                }
                            }
                        }
                        if((flag_p==1) && (flag_m==1))
                        {
                            y_mov=(tmp[0]*tmp[3]+tmp[2]*tmp[1])/(tmp[1]+tmp[3]);
                            resultFrame_gpu[((j+y) * cols +i+x) * 3] = frame_gpu[((j-int(roundf(y_mov))+y) * cols +i+x) * 3];
                            resultFrame_gpu[((j+y) * cols +i+x) * 3 + 1] = frame_gpu[((j-int(roundf(y_mov))+y) * cols +i+x) * 3 + 1];
                            resultFrame_gpu[((j+y) * cols +i+x) * 3 + 2] = frame_gpu[((j-int(roundf(y_mov))+y) * cols +i+x) * 3 + 2];
                            break;
                        }
                    }
                }
                }
            }

__global__ void horizontalCorrection(int* resultFrame, int* frame, float* avg, float* upperCurve, float* lowerCurve, int PupilMovVec, int PupilSquaredRadius, int cols, int h_start, int h_end, int w_start, int w_end)
{
int j = w_start + threadIdx.x + blockDim.x * blockIdx.x;
int i = h_start + threadIdx.y + blockDim.y * blockIdx.y;

if(i >= h_start && i < h_end && j >= w_start && j < w_end)
{
    int startPoint_r;
    int startPoint_l;

    int curvePx_prev[2] = {0, 0};

    int curvePx[2];
    computeCurvePx(upperCurve, lowerCurve, i, curvePx_prev, curvePx);
    int O[2] = {int(avg[1]), int(avg[0] + PupilMovVec)};
    int circlePx[2];
    computeCirclePx(O, PupilSquaredRadius, i, curvePx_prev, circlePx); // 이동한 눈동자와 y좌표 같은 교점

    int upPt[2] = {j, i};
    int lowPt[2] = {j - int(PupilMovVec), i};
    if((IsUpperComparison(upperCurve, upPt) && IsLowerComparison(lowerCurve, upPt)) || (IsUpperComparison(upperCurve, lowPt) && IsLowerComparison(lowerCurve, lowPt)))
    {
        float dist = sqrtf((avg[0] - (j - int(PupilMovVec))) * (avg[0] - (j - int(PupilMovVec))) + (avg[1] - i) * (avg[1] - i));
        // 거리가 눈동자 중심이랑 반지름 이내이고 이동 후 점이 곡선 범위 안에 들어온 점만 이동시킴
        if(dist < PupilSquaredRadius && IsUpperComparison(upperCurve, lowPt) && IsLowerComparison(lowerCurve, lowPt) && (IsUpperComparison(upperCurve, upPt) && IsLowerComparison(lowerCurve, upPt)))
        {
            resultFrame[(i * cols + j) * 3] = frame[(i * cols + (j - int(PupilMovVec))) * 3];
            resultFrame[(i * cols + j) * 3 + 1] = frame[(i * cols + (j - int(PupilMovVec))) * 3 + 1];
            resultFrame[(i * cols + j) * 3 + 2] = frame[(i * cols + (j - int(PupilMovVec))) * 3 + 2];
        }
        else
        {
            if(PupilMovVec >= 0)
            {
                startPoint_r = curvePx[1] - int((curvePx[1] - circlePx[0])/2); // 이동 후 오른쪽 중간 흰자
                startPoint_l = curvePx[0] + int(((circlePx[1] - PupilMovVec) - curvePx[0])/2); // 이동 전 왼쪽 중간 흰자
            }
            else
            {
                startPoint_r = curvePx[1] - int((curvePx[1] - (circlePx[0] - PupilMovVec))/2); // 이동 전 오른쪽 중간 흰자
                startPoint_l = curvePx[0] + int((circlePx[1] - curvePx[0])/2); // 이동 후 왼쪽 중간 흰자
            }
            // 눈동자 오른쪽 흰자 보간
            if(j >= circlePx[0] && j < startPoint_r)
            {
                float ratio = (startPoint_r - j) / float(startPoint_r - circlePx[0]);
                int idx = int(startPoint_r - (startPoint_r - (circlePx[0] - PupilMovVec)) * ratio);
                resultFrame[(i * cols + j) * 3] = frame[(i * cols + idx) * 3];
                resultFrame[(i * cols + j) * 3 + 1] = frame[(i * cols + idx) * 3 + 1];
                resultFrame[(i * cols + j) * 3 + 2] = frame[(i * cols + idx) * 3 + 2];
            }
            // 눈동자 왼쪽 흰자 보간
            else if(j <= circlePx[1] && j > startPoint_l)
            {
                float ratio = (j - startPoint_l) / float(circlePx[1] - startPoint_l);
                int idx = int(startPoint_l + ((circlePx[1] - PupilMovVec) - startPoint_l) * ratio);
                resultFrame[(i * cols + j) * 3] = frame[(i * cols + idx) * 3];
                resultFrame[(i * cols + j) * 3 + 1] = frame[(i * cols + idx) * 3 + 1];
                resultFrame[(i * cols + j) * 3 + 2] = frame[(i * cols + idx) * 3 + 2];
            }
        }
    }
}
}

__global__ void smooth(int* resultFrame, float* upperCurve, float* lowerCurve, int w_start, int w_end, int cols)
{
int j = w_start + threadIdx.x + blockDim.x * blockIdx.x;
// 경계 조금 부드럽게
if(j >= w_start && j <  w_end + 1)
{
    int testPt[2];
    int testPt2[2];
    computeCurvePt(upperCurve, j, testPt);
    computeCurvePt(lowerCurve, j, testPt2);
    resultFrame[(testPt[1] * cols + testPt[0]) * 3] = int(resultFrame[(testPt[1] * cols + testPt[0] - 1) * 3] / 2 + resultFrame[(testPt[1] * cols + testPt[0] + 1) * 3] / 2);
    resultFrame[(testPt[1] * cols + testPt[0]) * 3 + 1] = int(resultFrame[(testPt[1] * cols + testPt[0] - 1) * 3 + 1] / 2 + resultFrame[(testPt[1] * cols + testPt[0] + 1) * 3 + 1] / 2);
    resultFrame[(testPt[1] * cols + testPt[0]) * 3 + 2] = int(resultFrame[(testPt[1] * cols + testPt[0] - 1) * 3 + 2] / 2 + resultFrame[(testPt[1] * cols + testPt[0] + 1) * 3 + 2] / 2);
    resultFrame[(testPt2[1] * cols + testPt2[0]) * 3] = int(resultFrame[(testPt2[1] * cols + testPt2[0] - 1) * 3] / 2 + resultFrame[(testPt2[1] * cols + testPt2[0] + 1) * 3] / 2);
    resultFrame[(testPt2[1] * cols + testPt2[0]) * 3 + 1] = int(resultFrame[(testPt2[1] * cols + testPt2[0] - 1) * 3 + 1] / 2 + resultFrame[(testPt2[1] * cols + testPt2[0] + 1) * 3 + 1] / 2);
    resultFrame[(testPt2[1] * cols + testPt2[0]) * 3 + 2] = int(resultFrame[(testPt2[1] * cols + testPt2[0] - 1) * 3 + 2] / 2 + resultFrame[(testPt2[1] * cols + testPt2[0] + 1) * 3 + 2] / 2);
}
}
""",'nvcc') # 상응하는 Cuda C코드 작성 / 오류 없으면 코드 컴파일되어 장치에 로드
## 눈 검출 Left, Right 프레임별 위치
eyeDlibPtL =[]
eyeDlibPtR =[]

# 저장 프레임 수
frameSave=20

# 추가
phi = 0 # 5(기본 비슷) ~ -5
theta = 0.45
w_gaze = 0.0

cap = cv2.VideoCapture('vd5.mp4')    #960,720

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))

w_r=15
h_r=15

startflag=0 #시작flag(초기값 설정)
warpflag_prev=0 #이전프레임 warp 여부 1:wapred 2:original

prevTime = 0
##################tmp for frame rate up conversion

pad=50
################################################
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 학습모델 로드

detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1('shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 학습모델 로드

# range는 끝값이 포함안/특징마다 번호 부여
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))

index = LEFT_EYE + RIGHT_EYE

PupilMovVec_L = 0
PupilMovVec_R = 0
PupilSquaredRadius = 10
preAvgLx = 0
preAvgLy = 0
preAvgRx = 0
preAvgRy = 0

mvinfo_prev=[]
plist_prev=[]
eyeDlibPtL_prev = []
eyeDlibPtR_prev = []
region_prev=[10,10,10,10]

def pointExtraction(frame,gray,detector,predictor, eyeDlibPtL, eyeDlibPtR):
    frame_downsampled = frame.copy()
    gray_downsampled = gray.copy()
    frame_downsampled = cv2.resize(frame_downsampled,dsize=(int(gray.shape[1]*0.5), int(gray.shape[0]*0.5)),interpolation=cv2.INTER_AREA) #for dlib scale
    gray_downsampled = cv2.resize(gray_downsampled,dsize=(int(gray.shape[1]*0.5), int(gray.shape[0]*0.5)),interpolation=cv2.INTER_AREA) #for dlib scale

    dets = detector(gray_downsampled, 1) # 업셈플링 횟수

    eyeDlibPtL.clear()
    eyeDlibPtR.clear()
    list_points = [] # detect되는 얼굴 1개라고 가정...*
    for face in dets:
        shape = predictor(frame_downsampled, face) #얼굴에서 68개 점 찾기
        for p in shape.parts():
            list_points.append([p.x*2, p.y*2])

        list_points = np.array(list_points) # 리스트를 numpy로 변형
        cnt=0 #임시로 일단 표시하려고
        for i in list_points:
        #cv2.circle(frame, (i[0],i[1]), 2, (0, 255, 0), -1)
        #cv2.putText(frame, str(cnt),(i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
            cnt=cnt+1
        for idx in range(36,42):
            eyeDlibPtL.append(list_points[idx])
        #cv2.circle(frame, (list_points[idx][0], list_points[idx][1]), 2, (0, 255, 0), -1)
        for idx in range(42, 48):
            eyeDlibPtR.append(list_points[idx])
        #cv2.circle(frame, (list_points[idx][0], list_points[idx][1]), 2, (0, 255, 0), -1)

    return list_points

##곡선 바꾸기
#def CreateVeg(x_l,y_l,w,h): # 삭제예정
#    Crv = [0]*(640)
#    ax=0
#    ay=0

#    for t in range(0,w+10):
#        ax= int(  
#            (1-t/(w+9)) * (1-t/(w+9)) * (x_l) +
#              (2 * (t/(w+9)) * (1-t/(w+9)) * (x_l+w*6/10)) + 
#              (t/(w+9)) * (t/(w+9)) * (x_l+w))
#        ay=int(  
#            (1-t/(w+9)) * (1-t/(w+9)) * (int(y_l*1.04)) +
#              (2 * (t/(w+9)) * (1-t/(w+9)) * (y_l-h/4+h/8+h/16)) + 
#              (t/(w+9)) * (t/(w+9)) * (y_l+h/2))

#        Crv[ax]=ay


#    return Crv

def getTheta(tilt, p_eye,f): #
# in case ) tilt = 0 -> theta = p_eye/f
    theta = (
        math.atan(
            (p_eye*math.cos(tilt) - f*math.sin(tilt))/
            (p_eye*math.sin(tilt) + f*math.cos(tilt))
            )
            )
    return theta

def getRadius(Zn,faceHalf):
    return (Zn * Zn + pow(faceHalf, 2)) / (2* Zn)+5

def warping(phi,x,y,w,h,cy, Crv, frame_gpu, resultFrame_gpu, MaskFrame_gpu, w_gaze, cols, rows):

###Matching Z value using Cylinder Model
    f=int(64000/w)
#f=500
    w=int(w*1.2) # 상수 비율 / 어짜피 곡선으로 자를거면 크게 하던가 or 사용자 처음때 입력.
    x=int(x-w/5)
    h=int(h*2.5)
    y=int(y-h/2)

    mov_info=[0]*(int(w*2))*(int(h*2)) #변환 이후 정보 저장 배열 , 새로생성

    ZMatrix = np.empty((h, int(w*1.1))) #Z값 LUT

#FaceOrigin = ( h/2, w/2 ) #임시값 #이유 : ZMatrix는 눈영역만 따로만듬.
    FaceOrigin = np.array( (h/2, w/2) )

    faceHalf = w / 2 # 임시값

    Zn = faceHalf # 임시값

###
#Crv = CreateVeg(x,y,w,h) # 다른 곡선으로 바꿀예정

###
    theta = np.float32(getTheta(phi,cy,f))
    r = getRadius(Zn,faceHalf) # Cylinder model 반지름

    x_eye=x
    y_eye=y

# pyCuda
    mov_info_np = np.array(mov_info, dtype = np.int32)
    mov_info_gpu = cuda.mem_alloc(mov_info_np.nbytes)
    Crv_np = np.array(Crv, dtype = np.int32)
    Crv_gpu = cuda.mem_alloc(Crv_np.nbytes)
    ZMatrix = ZMatrix.astype(np.float32)
    ZMatrix_gpu = cuda.mem_alloc(ZMatrix.nbytes)

    cuda.memcpy_htod(mov_info_gpu, mov_info_np)
    cuda.memcpy_htod(Crv_gpu, Crv_np)
    cuda.memcpy_htod(ZMatrix_gpu, ZMatrix)


    func = mod.get_function("warping")
    bdim = (32, 32, 1)
    dx, mx = divmod(w, bdim[0])
    dy, my = divmod(h, bdim[1])
    gdim = (dx + (mx > 0), dy + (my > 0))
    func(frame_gpu, ZMatrix_gpu, mov_info_gpu, Crv_gpu, resultFrame_gpu, MaskFrame_gpu, np.float32(FaceOrigin[1]), np.int32(x_eye), np.int32(y_eye), np.float32(Zn), np.float32(r), np.float32(theta), np.int32(h), np.int32(w), np.int32(len(ZMatrix[0])), np.int32(cols), np.int32(rows), np.float32(w_gaze), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()

    func =  mod.get_function("interpolation")
    bdim = (32, 32, 1)
    dx, mx = divmod(int(h-1), bdim[0])
    dy, my = divmod(int(w-1), bdim[1])
    gdim = (dx + (mx > 0), dy + (my > 0))

    func(frame_gpu, mov_info_gpu, resultFrame_gpu, MaskFrame_gpu, np.int32(x), np.int32(y), np.int32(h), np.int32(w), np.int32(cols), np.int32(rows), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()

    return resultFrame_gpu

def horizontalCorrection(resultFrame, frame_gpu, eyeDlibPt, avg, PupilMovVec, PupilSquaredRadius, upperCurve, lowerCurve):

    #임시로넣음
    cols=640

    h_start = int((eyeDlibPt[1][1] + eyeDlibPt[2][1])/2 - 2)
    h_end = int((eyeDlibPt[4][1] + eyeDlibPt[5][1])/2 + 3)
    w_start = int(eyeDlibPt[0][0])
    w_end = int(eyeDlibPt[3][0])

    resultFrame = resultFrame.astype(np.uint32)
    avg = avg.astype(np.float32)
    upperCurve = upperCurve.astype(np.float32)
    lowerCurve = lowerCurve.astype(np.float32)

    resultFrame_gpu = cuda.mem_alloc(resultFrame.nbytes)
    avg_gpu = cuda.mem_alloc(avg.nbytes)
    upperCurve_gpu = cuda.mem_alloc(upperCurve.nbytes)
    lowerCurve_gpu = cuda.mem_alloc(lowerCurve.nbytes)

    cuda.memcpy_htod(resultFrame_gpu, resultFrame)
    cuda.memcpy_htod(avg_gpu, avg)
    cuda.memcpy_htod(upperCurve_gpu, upperCurve)
    cuda.memcpy_htod(lowerCurve_gpu, lowerCurve)

    func = mod.get_function("horizontalCorrection")

    w = w_end - w_start
    h = h_end - h_start

    bdim = (32, 32, 1)
    dx, mx = divmod(w, bdim[0])
    dy, my = divmod(h, bdim[1])
    gdim = (dx + (mx>0), dy + (my>0))
    func(resultFrame_gpu, frame_gpu, avg_gpu, upperCurve_gpu, lowerCurve_gpu, np.int32(PupilMovVec), np.int32(PupilSquaredRadius), np.int32(cols), np.int32(h_start), np.int32(h_end), np.int32(w_start), np.int32(w_end), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()

    func = mod.get_function("smooth")

    w = w_end - w_start

    bdim = (32, 1, 1)
    dx, mx = divmod(w, bdim[0])
    dy, my = divmod(1, bdim[1])
    gdim = (dx + (mx>0), dy + (my>0))
    func(resultFrame_gpu, upperCurve_gpu, lowerCurve_gpu, np.int32(w_start), np.int32(w_end), np.int32(cols), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()

    cuda.memcpy_dtoh(resultFrame, resultFrame_gpu)
    resultFrame = resultFrame.astype(np.uint8)
    return resultFrame

# 눈동자 색으로 눈동자 중심점 검출
def detectPupilCenter(frame_gpu, eyeDlibPtL, eyeDlibPtR, cols, preAvgLx, preAvgLy, preAvgRx, preAvgRy, upperCurve_l, lowerCurve_l, upperCurve_r, lowerCurve_r):
    PupilLocationLx = np.array([0] * (int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2) - int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2)) * (eyeDlibPtL[3][0] - eyeDlibPtL[0][0]), dtype = np.int32)
    PupilLocationLy = np.array([0] * (int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2) - int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2)) * (eyeDlibPtL[3][0] - eyeDlibPtL[0][0]), dtype = np.int32)
    PupilLocationLx_gpu = cuda.mem_alloc(PupilLocationLx.nbytes)
    PupilLocationLy_gpu = cuda.mem_alloc(PupilLocationLy.nbytes)

    PupilLocationRx = np.array([0] * (int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2) - int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2)) * (eyeDlibPtR[3][0] - eyeDlibPtR[0][0]), dtype = np.int32)
    PupilLocationRy = np.array([0] * (int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2) - int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2)) * (eyeDlibPtR[3][0] - eyeDlibPtR[0][0]), dtype = np.int32)

    PupilLocationRx_gpu = cuda.mem_alloc(PupilLocationRx.nbytes)
    PupilLocationRy_gpu = cuda.mem_alloc(PupilLocationRy.nbytes)

    cuda.memcpy_htod(PupilLocationLx_gpu, PupilLocationLx)
    cuda.memcpy_htod(PupilLocationLy_gpu, PupilLocationLy)
    cuda.memcpy_htod(PupilLocationRx_gpu, PupilLocationRx)
    cuda.memcpy_htod(PupilLocationRy_gpu, PupilLocationRy)

    upperCurve_l = upperCurve_l.astype(np.float32)
    lowerCurve_l = lowerCurve_l.astype(np.float32)
    upperCurve_r = upperCurve_r.astype(np.float32)
    lowerCurve_r = lowerCurve_r.astype(np.float32)

    upperCurve_l_gpu = cuda.mem_alloc(upperCurve_l.nbytes)
    lowerCurve_l_gpu = cuda.mem_alloc(lowerCurve_l.nbytes)
    upperCurve_r_gpu = cuda.mem_alloc(upperCurve_r.nbytes)
    lowerCurve_r_gpu = cuda.mem_alloc(lowerCurve_r.nbytes)

    cuda.memcpy_htod(upperCurve_l_gpu, upperCurve_l)
    cuda.memcpy_htod(lowerCurve_l_gpu, lowerCurve_l)
    cuda.memcpy_htod(upperCurve_r_gpu, upperCurve_r)
    cuda.memcpy_htod(lowerCurve_r_gpu, lowerCurve_r)

    func = mod.get_function("pupilCheckL")
    bdim = (32, 32, 1)
    temph = int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2) - int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2)
    tempw = int(eyeDlibPtL[3][0] - eyeDlibPtL[0][0])
    dy, my = divmod(int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2) - int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2), bdim[0])
    dx, mx = divmod(int(eyeDlibPtL[3][0] - eyeDlibPtL[0][0]), bdim[1])
    gdim = (dx + (mx>0), dy + (my>0))
    func(PupilLocationLx_gpu, PupilLocationLy_gpu, frame_gpu, upperCurve_l_gpu, lowerCurve_l_gpu, np.int32(eyeDlibPtL[0][0]), np.int32((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2), np.int32(eyeDlibPtL[3][0] - eyeDlibPtL[0][0]), np.int32(cols), np.int32(tempw), np.int32(temph), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()

    cuda.memcpy_dtoh(PupilLocationLx, PupilLocationLx_gpu)
    cuda.memcpy_dtoh(PupilLocationLy, PupilLocationLy_gpu)
    func = mod.get_function("pupilCheckR")
    bdim = (32, 32, 1)
    temph = int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2) - int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2)
    tempw = int(eyeDlibPtR[3][0] - eyeDlibPtR[0][0])
    dy, my = divmod(int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2) - int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2), bdim[0])
    dx, mx = divmod(int(eyeDlibPtR[3][0] - eyeDlibPtR[0][0]), bdim[1])
    gdim = (dx + (mx>0), dy + (my>0))
    func(PupilLocationRx_gpu, PupilLocationRy_gpu, frame_gpu, upperCurve_r_gpu, lowerCurve_r_gpu, np.int32(eyeDlibPtR[0][0]), np.int32((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2), np.int32(eyeDlibPtR[3][0] - eyeDlibPtR[0][0]), np.int32(cols), np.int32(tempw), np.int32(temph), block = bdim, grid = gdim)
    pycuda.driver.Context.synchronize()
    cuda.memcpy_dtoh(PupilLocationRx, PupilLocationRx_gpu)
    cuda.memcpy_dtoh(PupilLocationRy, PupilLocationRy_gpu)

    if len(PupilLocationLx.nonzero()[0]) != 0 and len(PupilLocationLy.nonzero()[0]) != 0 and len(PupilLocationRx.nonzero()[0]) != 0 and len(PupilLocationRy.nonzero()[0]) != 0:
        avgLx = sum(PupilLocationLx) / len(PupilLocationLx.nonzero()[0])
        avgLy = sum(PupilLocationLy) / len(PupilLocationLy.nonzero()[0])
        avgRx = sum(PupilLocationRx) / len(PupilLocationRx.nonzero()[0])
        avgRy = sum(PupilLocationRy) / len(PupilLocationRy.nonzero()[0])
        (preAvgLx, preAvgLy, preAvgRx, preAvgRy) = (avgLx, avgLy, avgRx, avgRy)
    else:
        (avgLx, avgLy, avgRx, avgRy) = (preAvgLx, preAvgLy, preAvgRx, preAvgRy)

    return (avgLx, avgLy, avgRx, avgRy, preAvgLx, preAvgLy, preAvgRx, preAvgRy, frame_gpu)

def computeCurve(p0, p1, p2):
    A = np.array([[p0[0] * p0[0], p0[0], 1], [p1[0] * p1[0], p1[0], 1], [p2[0] * p2[0], p2[0], 1]])
    B = np.array([p0[1], p1[1], p2[1]])
    return np.linalg.solve(A, B)

#def computeCurvePt(curveP, pt):
#    resultPt =  np.array([pt, int(curveP[0] * pt * pt + curveP[1] * pt + curveP[2])])
#    return resultPt

def setCurvePt(curveP, x, W, cols, add):
    Crv = [0]*cols
    for i in range(x, x + W):
        Crv[i] =  int(curveP[0] * i * i + curveP[1] * i + curveP[2] - add)
    return Crv

# 근의 공식 이용해서 y좌표 주어졌을 때 곡선과 만나는 x좌표 찾기 
def computeCurvePx(UpperCurve, LowerCurve, py, curvePx_prev):
    if UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py) < 0 or LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py)  < 0:
        return curvePx_prev
    else:
        x1_Up = int((-UpperCurve[1] - math.sqrt(UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py))) / (2 * UpperCurve[0]))
        x2_Up = int((-UpperCurve[1] + math.sqrt(UpperCurve[1] * UpperCurve[1] - 4 * UpperCurve[0] * (UpperCurve[2] - py))) / (2 * UpperCurve[0]))
        x1_Low = int((-LowerCurve[1] + math.sqrt(LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py))) / (2 * LowerCurve[0]))
        x2_Low = int((-LowerCurve[1] - math.sqrt(LowerCurve[1] * LowerCurve[1] - 4 * LowerCurve[0] * (LowerCurve[2] - py))) / (2 * LowerCurve[0]))

        x1 = x1_Low if x1_Up < x1_Low else x1_Up
        x2 = x2_Up if x2_Low > x2_Up else x2_Low

        return np.array([x1, x2])

#def IsLowerComparison(curveP, pt):
#    if pt[1] < (curveP[0] * pt[0] * pt[0] + curveP[1] * pt[0] + curveP[2]):
#        return True
#    else:
#        return False

#def IsUpperComparison(curveP, pt):
#    if pt[1] > (curveP[0] * pt[0] * pt[0] + curveP[1] * pt[0] + curveP[2]):
#        return True
#    else:
#        return False

# y좌표가 주어졌을 때 눈동자와 만나는 점 찾기 (O: 눈동자 원점, R: 반지름, py: 주어진 y좌표)
def computeCirclePx(O, R, py, circlePx_prev):
    if R * R - (py - O[0]) * (py - O[0]) < 0:
        return circlePx_prev
    else:
        x1 = int(math.sqrt(R * R - (py - O[0]) * (py - O[0])) + O[1]) # 큰 x
        x2 = int(-math.sqrt(R * R - (py - O[0]) * (py - O[0])) + O[1]) # 작은 x
        return np.array([x1, x2])

########################## Main ############################


#if cap.isOpened():
#    ret, frame_prev=cap.read()
#    frame_prev = cv2.resize(frame_prev,(640, 480),interpolation=cv2.INTER_AREA)
#    rows, cols = frame_prev.shape[:2]
#    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
#    frame_prev = cv2.warpAffine(frame_prev, rotation_matrix, (cols, rows))
#    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)

#    Result_prev=frame_prev.copy()

########## Cam Loop #####################################################################################################################################################################################
#while(cap.isOpened()):
class MainRPC(object):
    def mainCorrection(frame):
        cap = cv2.VideoCapture("testvideo.mp4")\

#####################################################3
        eyeDlibPtL = []
        eyeDlibPtR = []

        # # 저장 프레임 수
        # frameSave = 20
        #
        # # 추가
        # phi = 0  # 5(기본 비슷) ~ -5

        theta = 0.45
        w_gaze = 0.0
        #
        # cap = cv2.VideoCapture('vd5.mp4')  # 960,720
        #
        # RIGHT_EYE = list(range(36, 42))
        # LEFT_EYE = list(range(42, 48))
        #
        # w_r = 15
        # h_r = 15
        #
        # startflag = 0  # 시작flag(초기값 설정)
        # warpflag_prev = 0  # 이전프레임 warp 여부 1:wapred 2:original
        #
        # prevTime = 0
        # ##################tmp for frame rate up conversion
        #
        # pad = 50
        # ################################################
        #
        # detector = dlib.get_frontal_face_detector()
        # # detector = dlib.cnn_face_detection_model_v1('shape_predictor_68_face_landmarks.dat')
        # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 학습모델 로드
        #
        # # range는 끝값이 포함안/특징마다 번호 부여
        # RIGHT_EYE = list(range(36, 42))
        # LEFT_EYE = list(range(42, 48))
        #
        # index = LEFT_EYE + RIGHT_EYE

        PupilMovVec_L = 0
        PupilMovVec_R = 0
        PupilSquaredRadius = 10
        preAvgLx = 0
        preAvgLy = 0
        preAvgRx = 0
        preAvgRy = 0

        # mvinfo_prev = []
        # plist_prev = []
        # eyeDlibPtL_prev = []
        # eyeDlibPtR_prev = []
        # region_prev = [10, 10, 10, 10]
        #################################################
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


        # range는 끝값이 포함안/특징마다 번호 부여
        RIGHT_EYE = list(range(36, 42))
        LEFT_EYE = list(range(42, 48))

        index = LEFT_EYE + RIGHT_EYE



        ########## Cam Loop#####################################################################################################################################################################################
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame,(640, 480),interpolation=cv2.INTER_AREA)
                rows, cols = frame.shape[:2]

                rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 270, 1)
                #frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))
                gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                gray = cv2.blur(gray, (3, 3), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
                gray = cv2.equalizeHist(gray)

                # 1. Eye Detecting
                #list_points = pointExtraction(frame,gray,detector,predictor)
                list_points = pointExtraction(frame,gray,detector,predictor, eyeDlibPtL, eyeDlibPtR)

                ###################################################################################

                ResultFrame=frame.copy()
                ResultFrame_h=frame.copy()
                TrackingFrame = frame.copy() # Warping 결과 Frame
                MaskFrame=frame.copy() # Warping 값 flag 배열(for interpolation)

                #detected_f = faces.detectMultiScale(gray, 1.3, 5) #Face(detected)
                #detected = eyes.detectMultiScale(gray, 1.3, 5) #Eyes(detected)
                warpflag=1 #warp초기값
                if len(list_points)<68 or abs(eyeDlibPtL[0][1] - eyeDlibPtL[3][1]) > (eyeDlibPtL[3][0] - eyeDlibPtL[0][0]) / 2 or abs(eyeDlibPtR[0][1] - eyeDlibPtR[3][1]) > (eyeDlibPtR[3][0] - eyeDlibPtR[0][0]) / 2 :
                    #wapredflag==0으로
                    print("warpedflag=0 or can't find eye or etc...")
                    #print(list_points)
                    warpflag=0


                # 2. Gaze Estimation
                    #gazeVec = ExtractGazeVector() #작성 예정

                if startflag == 0:
                    print("start")

                    startflag = 1
                else:
                    if warpflag==1 :

                        #warpflag_prev 판단

                        #cx,cy=tuple(np.average(eyeDlibPtL,0))
                        #cx_r,cy_r=tuple(np.average(eyeDlibPtR,0))
                        #cv2.circle(frame,(int(cx),int(cy)),2,(0,0,255),-1)
                        #cv2.circle(frame,(int(cx_r),int(cy_r)),2,(0,0,255),-1)

                        ############################################

                        #left eye
                        #x=list_points[36][0]
                        #y=int((list_points[37][1]+list_points[38][1])/2)
                        #w=int((list_points[39][0]-x)*1.5)
                        #h=int((list_points[41][1]-y)*1.5)

                        x=list_points[36][0]
                        y=int((list_points[37][1]+list_points[38][1])/2)
                        w=int((list_points[39][0]-x)*1.1)
                        h=int((list_points[41][1]-y)*1.5)

                        #right eye
                        x_r=list_points[42][0]
                        y_r=int((list_points[43][1]+list_points[44][1])/2)
                        w_r=int((list_points[45][0]-x_r)*1.5)
                        h_r=int((list_points[47][1]-y_r)*1.5)

                        PupilSquaredRadius = int((eyeDlibPtL[3][0] - eyeDlibPtL[0][0])/3.3)
                        ######################### tracking 안정화 #############################
                        diffsum = 999
                        x_p,y_p,w_p,h_p = region_prev

                        for i in range(y_p, y_p + h_p):
                            for j in range(x_p, x_p + w_p):
                                diffsum = diffsum + abs(int(gray[i][j]) - int(gray_prev[i][j]))
                        diffsum = diffsum/(w_p * h_p)

                        if diffsum < 5 and len(plist_prev) != 0 and len(eyeDlibPtL_prev) != 0 and len(eyeDlibPtR_prev) != 0:
                            #print("not move")
                            list_points = plist_prev
                            eyeDlibPtL = eyeDlibPtL_prev.copy()
                            eyeDlibPtR  = eyeDlibPtR_prev.copy()

                        #for i in list_points:
                        #    cv2.circle(frame, (i[0],i[1]), 2, (0, 255, 0), -1)
                        #for i in eyeDlibPtL:
                        #    cv2.circle(frame, (i[0],i[1]), 2, (0, 255, 0), -1)
                        #for i in eyeDlibPtR:
                        #    cv2.circle(frame, (i[0],i[1]), 2, (0, 255, 0), -1)

                        region_prev = [x,y,w,h]
                        ##########################################
                        h_start_r = int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2 - 2)
                        h_end_r = int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2 + 2)
                        w_start_r = eyeDlibPtR[0][0] + 2
                        w_end_r = eyeDlibPtR[3][0] + 3
                        h_start_l = int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2 - 4)
                        h_end_l = int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2 + 2)
                        w_start_l = eyeDlibPtL[0][0]
                        w_end_l = eyeDlibPtL[3][0] + 3

                        upperCurve_l = computeCurve((w_start_l, eyeDlibPtL[0][1] - 3), (int((eyeDlibPtL[1][0] + eyeDlibPtL[2][0])/2), h_start_l), (w_end_l, eyeDlibPtL[3][1]))
                        lowerCurve_l = computeCurve((w_start_l , eyeDlibPtL[0][1]),(int((eyeDlibPtL[4][0] + eyeDlibPtL[5][0])/2), h_end_l), (w_end_l, eyeDlibPtL[3][1]))
                        upperCurve_r = computeCurve((w_start_r, eyeDlibPtR[0][1] - 3), (int((eyeDlibPtR[1][0] + eyeDlibPtR[2][0])/2), h_start_r), (w_end_r, eyeDlibPtR[3][1]))
                        lowerCurve_r = computeCurve((w_start_r , eyeDlibPtR[0][1]),(int((eyeDlibPtR[4][0] + eyeDlibPtR[5][0])/2), h_end_r), (w_end_r, eyeDlibPtR[3][1]))

                        frame = frame.astype(np.uint32)
                        frame_gpu = cuda.mem_alloc(frame.nbytes)
                        cuda.memcpy_htod(frame_gpu, frame)
                        # 눈동자 색 눈동자 중심 검출
                        (avgLx, avgLy, avgRx, avgRy, preAvgLx, preAvgLy, preAvgRx, preAvgRy, frame_gpu) = detectPupilCenter(frame_gpu, eyeDlibPtL, eyeDlibPtR, cols, preAvgLx, preAvgLy, preAvgRx, preAvgRy, upperCurve_l, lowerCurve_l, upperCurve_r, lowerCurve_r)
                        #cv2.circle(ResultFrame, (int(avgLx), int(avgLy)), 3, (0,0,255), -1)
                        #cv2.circle(ResultFrame, (int(avgRx), int(avgRy)), 3, (0,0,255), -1)
                        ##########################################

                        # 곡선 계산
                        tempwL = eyeDlibPtL[3][0] - eyeDlibPtL[0][0]
                        tempwR = eyeDlibPtR[3][0] - eyeDlibPtR[0][0]
                        add = 15 # 눈매 위 어디까지 교정
                        # 와핑 위한 곡선 눈썹이랑 눈 중간 정도
                        CrvL = setCurvePt(computeCurve(eyeDlibPtL[0], (int((eyeDlibPtL[1][0] + eyeDlibPtL[2][0])/2), int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2)), eyeDlibPtL[3]), x - int(tempwL * 0.3), int(tempwL * 1.6), cols, add)
                        CrvR = setCurvePt(computeCurve(eyeDlibPtR[0], (int((eyeDlibPtR[1][0] + eyeDlibPtR[2][0])/2), int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2)), eyeDlibPtR[3]), x_r - int(tempwR * 0.3), int(tempwR * 1.6), cols, add)

                        ##5. Left-Right Correction
                        avgL = np.array((avgLx, avgLy))
                        avgR = np.array((avgRx, avgRy))

                        tempEyeW_L = eyeDlibPtL[3][0] - eyeDlibPtL[0][0]
                        tempEyeW_R = eyeDlibPtR[3][0] - eyeDlibPtR[0][0]

                        # 눈 안감았을 때만 처리
                        if eyeDlibPtL[5][1] - eyeDlibPtL[1][1] > tempEyeW_L * 0.2:
                        #    # 눈동자가 어느 정도 범위 안에 들어와야 교정
                        #    #if avgL[0] > eyeDlibPtL[0][0] + tempEyeW_L * 2 / 7 and avgL[0] < eyeDlibPtL[0][0] + tempEyeW_L * 5 / 7 and avgR[0] > eyeDlibPtR[0][0] + tempEyeW_R * 2 / 7 and avgR[0] < eyeDlibPtR[0][0] + tempEyeW_R * 5 / 7:
                        #    ResultFrame = horizontalCorrection(ResultFrame, frame, eyeDlibPtL, avgL, PupilMovVec_L, PupilSquaredRadius)
                        #    ResultFrame = horizontalCorrection(ResultFrame, frame, eyeDlibPtR, avgR, PupilMovVec_R, PupilSquaredRadius)
                                #h_start_r = int((eyeDlibPtR[1][1] + eyeDlibPtR[2][1])/2 - 2)
                                #h_end_r = int((eyeDlibPtR[4][1] + eyeDlibPtR[5][1])/2 + 2)
                                #w_start_r = eyeDlibPtR[0][0] + 2
                                #w_end_r = eyeDlibPtR[3][0] + 3
                                #h_start_l = int((eyeDlibPtL[1][1] + eyeDlibPtL[2][1])/2 - 4)
                                #h_end_l = int((eyeDlibPtL[4][1] + eyeDlibPtL[5][1])/2 + 2)
                                #w_start_l = eyeDlibPtL[0][0]
                                #w_end_l = eyeDlibPtL[3][0] + 3

                                #upperCurve_l = computeCurve((w_start_l, eyeDlibPtL[0][1] - 3), (int((eyeDlibPtL[1][0] + eyeDlibPtL[2][0])/2), h_start_l), (w_end_l, eyeDlibPtL[3][1]))
                                #lowerCurve_l = computeCurve((w_start_l , eyeDlibPtL[0][1]),(int((eyeDlibPtL[4][0] + eyeDlibPtL[5][0])/2), h_end_l), (w_end_l, eyeDlibPtL[3][1]))
                                #upperCurve_r = computeCurve((w_start_r, eyeDlibPtR[0][1] - 3), (int((eyeDlibPtR[1][0] + eyeDlibPtR[2][0])/2), h_start_r), (w_end_r, eyeDlibPtR[3][1]))
                                #lowerCurve_r = computeCurve((w_start_r , eyeDlibPtR[0][1]),(int((eyeDlibPtR[4][0] + eyeDlibPtR[5][0])/2), h_end_r), (w_end_r, eyeDlibPtR[3][1]))

                                circlePx_prev = np.array((0, 0))
                                curvePx_prev = np.array((0, 0))

                                curvePx_avg_L = computeCurvePx(upperCurve_l, lowerCurve_l, avgL[1], curvePx_prev)
                                circlePx_avg_L = computeCirclePx((avgL[1], avgL[0]), PupilSquaredRadius, avgL[1], circlePx_prev)
                                curvePx_avg_R = computeCurvePx(upperCurve_r, lowerCurve_r, avgR[1], curvePx_prev)
                                circlePx_avg_R = computeCirclePx((avgR[1], avgR[0]), PupilSquaredRadius, avgR[1], circlePx_prev)

                                # 눈동자가 좌우 눈매에 닿으면 좌우 교정X
                                if circlePx_avg_L[0] > curvePx_avg_L[1] or circlePx_avg_L[1] < curvePx_avg_L[0] or circlePx_avg_R[0] > curvePx_avg_R[1] or circlePx_avg_R[1] < curvePx_avg_R[0]:
                                    print("No Horizental Correction")
                                else:
                                    ResultFrame = horizontalCorrection(ResultFrame, frame_gpu, eyeDlibPtL, avgL, PupilMovVec_L, PupilSquaredRadius, upperCurve_l, lowerCurve_l)
                                    ResultFrame = horizontalCorrection(ResultFrame, frame_gpu, eyeDlibPtR, avgR, PupilMovVec_R, PupilSquaredRadius, upperCurve_r, lowerCurve_r)

                        ResultFrame = cv2.medianBlur(ResultFrame, 3)
                        ResultFrame_h = ResultFrame.copy()

                        ResultFrame = ResultFrame.astype(np.uint32)
                        ResultFrame_h = ResultFrame_h.astype(np.uint32)
                        MaskFrame = MaskFrame.astype(np.uint32)
                        ResultFrame_gpu = cuda.mem_alloc(ResultFrame.nbytes)
                        ResultFrame_h_gpu = cuda.mem_alloc(ResultFrame_h.nbytes)
                        MaskFrame_gpu = cuda.mem_alloc(MaskFrame.nbytes)
                        cuda.memcpy_htod(ResultFrame_gpu, ResultFrame)
                        cuda.memcpy_htod(ResultFrame_h_gpu, ResultFrame_h)
                        cuda.memcpy_htod(MaskFrame_gpu, MaskFrame)
                        #3. Warping_L
                        ResultFrame_h_gpu = warping(phi,x,y,w,h,avgL[1],CrvL, ResultFrame_gpu, ResultFrame_h_gpu, MaskFrame_gpu, w_gaze, cols, rows)
                        #4. Warping_R
                        ResultFrame_h_gpu = warping(phi,x_r,y_r,w_r,h_r,avgR[1],CrvR, ResultFrame_gpu, ResultFrame_h_gpu, MaskFrame_gpu, w_gaze, cols, rows)

                        cuda.memcpy_dtoh(ResultFrame_h, ResultFrame_h_gpu)
                        cuda.memcpy_dtoh(ResultFrame, ResultFrame_gpu)
                        cuda.memcpy_dtoh(frame, frame_gpu)
                        ResultFrame_h = ResultFrame_h.astype(np.uint8)
                        ResultFrame = ResultFrame.astype(np.uint8)
                        frame = frame.astype(np.uint8)
                        #cv2.circle(frame, (int(avgLx) + PupilSquaredRadius, int(avgLy)), 3, (0,0,255), -1)
                        #cv2.circle(frame, (int(avgRx) + PupilSquaredRadius, int(avgRy)), 3, (0,0,255), -1)
                    #############################################

                    else : # warpflag 0일 때
                        #warpflag_prev 판단
                        print("warpflag=0")

                plist_prev = list_points
                eyeDlibPtL_prev = eyeDlibPtL.copy()
                eyeDlibPtR_prev = eyeDlibPtR.copy()
                #gray_prev = gray.copy()
                #frame_prev=frame.copy() ## 이전프레임 frame rate up을 위한


                ResultFrame_h = cv2.medianBlur(ResultFrame_h, 3)
                #frame = cv2.medianBlur(frame, 3)
                curTime = time.time()
                sec = curTime - prevTime
                prevTime = curTime

                fps = 1 / sec
                print(fps)
                str_fps = "FPS : %0.1f" % fps
                cv2.putText(frame, str_fps, (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

                cv2.imshow('Frame',frame)
                cv2.imshow('ResultFrame',ResultFrame)
                cv2.imshow('ResultFrame_h',ResultFrame_h)

                key = cv2.waitKey(1)

                # '1' 누르면 phi 줄여서 교정 많이 되게 '2' 누르면 phi 높여서 기존에 가깝게 / 최대 최솟값 정해놓기
                if key == ord('1'):
                    if w_gaze > 0.0:
                        w_gaze -= 0.05
                elif key == ord('2'):
                    if w_gaze < 0.3:
                        w_gaze += 0.05
                elif key == ord('3'):
                    if PupilMovVec_L > -8:
                        PupilMovVec_L -= 1
                        PupilMovVec_R -= 1
                elif key == ord('4'):
                    if PupilMovVec_L < 8:
                        PupilMovVec_L += 1
                        PupilMovVec_R += 1

        #if key == ord('q'):
            #break

    #else:
    #    break

#cap.release()
    cv2.destroyAllWindows()








s = zerorpc.Server(MainRPC())
s.bind("tcp://*:4242")
s.run()

####################################################################################################################################################################################################