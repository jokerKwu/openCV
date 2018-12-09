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
//�� ���� �κ�
//////////////////////////////////////////////

#define SIZE 256
void calculateHisogram(int hist[256], Mat img);
int  basicGlobalThresholing(int hist[256]);
void toBinaryImage(Mat ori, Mat dst, int globalT);
unsigned char** allocMem(int height, int width);
void cutCircle(Mat img, int D);
Mat eyeDect(Mat faceDetected);


////////////////////////////////////////////// 
//�� ���� �κ�
////////////////////////////////////////////// 
static int height = 512; //resize ���� �̹��� ũ�� ����
static int width = 512; //resize ���� �̹��� ũ�� ����
bool cmp(Point a, Point b);
void harrisCorner(unsigned char** img, vector<Point>&corners, double th, int height, int width);      //�Ӱ谪�� ���� Ư¡���� �����ϴ� �Լ�   (�ʷϻ���)
vector<Point> centerValue(vector<vector<Point>>& c);                        // ������ �Ÿ��� ���� �з�ȭ�� Ư¡���� ���� �߽��� ���ϴ� �Լ�   (�Ķ�����)
Point faceCenter(vector<Point> a);                                       // �з�ȭ�� Ư¡���� ���� �߽ɰ��鰣�� �߽ɰ� ���ϴ� �Լ� (��������)
vector<vector<Point>> bfs(int maskSize, int clusterCnt, vector<Point> cList);      // Ư¡�� Ž���� ���ؼ� �з�ȭ ���ϴ� �Լ�
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