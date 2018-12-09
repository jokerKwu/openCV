#pragma once
#include"opencv2/opencv.hpp"
#include<iostream>
#include<vector>
#include<string.h>
#include<queue>
#include<algorithm>
#include<math.h>
#include <utility>
#include<string>
using namespace cv;
using namespace std;

//////////////////////////////////////////////
//눈 추출 부분
//////////////////////////////////////////////

#define SIZE 256
void calculateHisogram(int hist[256], Mat img);
int  basicGlobalThresholing(int hist[256]);
void toBinaryImage(Mat ori, Mat dst, int globalT);
unsigned char** allocMem(int height, int width);
void cutCircle(Mat img, int D);
Mat eyeDect(Mat faceDetected);


////////////////////////////////////////////// 
//얼굴 추출 부분
////////////////////////////////////////////// 
static int height = 512; //resize 영상 이미지 크기 설정
static int width = 512; //resize 영상 이미지 크기 설정
bool cmp(Point a, Point b);
void harrisCorner(unsigned char** img, vector<Point>&corners, double th, int height, int width);      //임계값을 통한 특징점을 추출하는 함수   (초록색점)
vector<Point> centerValue(vector<vector<Point>>& c);                        // 임의의 거리에 따른 분류화된 특징점들 간에 중심점 구하는 함수   (파랑색점)
Point faceCenter(vector<Point> a);                                       // 분류화된 특징점들 간에 중심값들간에 중심값 구하는 함수 (빨간색점)
vector<vector<Point>> bfs(int maskSize, int clusterCnt, vector<Point> cList);      // 특징점 탐색을 통해서 분류화 구하는 함수
Point rectLength(Point a, vector<Point> b);
Mat imgDisplay(int pointDist, int keyPointCnt, int th, Mat ori_img, int count, int height, int width, int WhiteCenter);
void pointPrint(int x, int y, int color[3], Mat img);
double** MemAlloc2D(int nHeight, int nWidth, unsigned char nInitVal);
Mat faceDetected(Mat img, int startX, int startY, int destX, int destY);
int ColorSlicing(Mat inputImg, int name);
int dx[3] = { -1,0,1 };
int dy[3] = { -1,0,1 };
Mat hlsImg, skinImg;
IplImage* convertImageHSVtoRGB(const IplImage *imageHSV);
IplImage* convertImageRGBtoHSV(const IplImage *imageRGB);
void mophologyErosion(Mat result, Mat copied);
void mophologyDilation(Mat result, Mat copied);
int** makeMeanOfCell(Mat result);
void copyMatToArr(Mat from, unsigned char** to, int height, int width);
Mat Erosion(Mat img);
Mat Dilation(Mat img);