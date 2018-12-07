#include"project.h"

int main() { 
	int pointDist = 5, keyPointCnt = 20;   // pointDist 마스크 길이, keyPointCnt 밀도 개수
	double th = 1000.0;//임계값
	string imgPathSetting = "C:\\Users\\joon\\Desktop\\1\\";//이미지 경로 설정 이미지가 들어있는 폴더 경로로 변경
	int startNum, endNum;
	cout << "####현재 이미지 경로 설정 상태#####" << '\n';
	cout << imgPathSetting + "?.png\n\n\n";
	cout << "사진 시작 번호 >>"; cin >> startNum;
	cout << "사진 끝 번호 >>"; cin >> endNum;
	for (int i = startNum; i <= endNum; i++) {
		string imgPath = imgPathSetting + to_string(i) + ".png";
		Mat filter_img = imread(imgPath);
		int WhiteCenter = ColorSlicing(filter_img,i);
		Mat faceDetected=imgDisplay(pointDist, keyPointCnt, th, filter_img, i, height, width, WhiteCenter);
		Mat eyeDectected=eyeDect(faceDetected);
		string faceName, eyeName;
		imshow(to_string(i)+"face", faceDetected);
		imshow(to_string(i)+"eye", eyeDectected);
	}
	waitKey(0);
}
Mat eyeDect(Mat faceDetected) {
	Mat img_resize(Size(faceDetected.cols, faceDetected.rows), CV_8UC1);

	Mat img_gray;
	resize(faceDetected, img_resize, Size(256, 256), 0, 0, CV_INTER_CUBIC);
	Mat result = Mat(256, 256, CV_8UC1);
	int hist[256] = { 0 };
	calculateHisogram(hist, img_resize);

	int globalT = basicGlobalThresholing(hist);

	toBinaryImage(img_resize, result, globalT);

	cutCircle(result, 110);

	int numberOfBlack = 0;

	int masksize = 20;
	unsigned char** mask = allocMem(masksize, masksize);
	int center = masksize / 2;
	for (int h = 0; h < masksize; h++) {
		for (int w = 0; w < masksize; w++) {
			if ((h - center)*(h - center) + (w - center)*(w - center) < 6 * 6) {
				mask[h][w] = 255;
			}
		}
	}



	for (int h = 128; h >= 0; h--) {
		numberOfBlack = 0;
		for (int i = h - 20; i < h; i++) {
			for (int w = 0; w < SIZE; w++) {
				if (result.at<uchar>(h, w) == 0) {
					numberOfBlack++;
				}
			}
		}
		if (numberOfBlack > 400) {
			for (int i = 0; i < SIZE; i++) {
				result.at<uchar>(h, i) = 128;
				result.at<uchar>(h + 20, i) = 128;

			}
			return result;
		}
	}
	return result;
}


int ColorSlicing(Mat inputImg, int name)
{
	int xpoint = 0;
	int ypoint = 0;
	int count = 0;

	skinImg = inputImg.clone();
	cvtColor(inputImg, hlsImg, CV_RGB2HLS);
	vector<Mat> hls_images(3);
	split(hlsImg, hls_images);

	for (int i = 0; i < hlsImg.rows; i++) {
		for (int j = 0; j < hlsImg.cols; j++) {
			unsigned char H = hlsImg.at<Vec3b>(i, j)[0];      //색상
			unsigned char L = hlsImg.at<Vec3b>(i, j)[1];      //채도
			unsigned char S = hlsImg.at<Vec3b>(i, j)[2];      //광도
			double LS_ratio = ((double)L) / ((double)S);
			bool skin_pixel = (S >= 50) && (LS_ratio > 0.5) && (LS_ratio < 3.0) && ((H <= 14) || ((H >= 100) && (H <= 130)));

			if (skin_pixel)
			{
				xpoint += j;
				ypoint += i;
				count++;
				skinImg.at<Vec3b>(i, j)[0] = 255;
				skinImg.at<Vec3b>(i, j)[1] = 255;
				skinImg.at<Vec3b>(i, j)[2] = 255;
			}
			else
			{
				skinImg.at<Vec3b>(i, j)[0] = 0;
				skinImg.at<Vec3b>(i, j)[1] = 0;
				skinImg.at<Vec3b>(i, j)[2] = 0;
			}

		}
	}


	skinImg.at<Vec3b>(ypoint / count, xpoint / count)[0] = 0;
	skinImg.at<Vec3b>(ypoint / count, xpoint / count)[1] = 0;
	skinImg.at<Vec3b>(ypoint / count, xpoint / count)[2] = 255;
	Mat fiter_img;
	resize(skinImg, fiter_img, Size(height, width), 0, 0, CV_INTER_CUBIC);
	//imshow(to_string(name + 100), fiter_img);

	int WhiteCenter = (((double)xpoint / (double)count) / (double)inputImg.cols) * width;


	return WhiteCenter;
}

Mat imgDisplay(int pointDist, int keyPointCnt, int th, Mat ori_img, int count, int height, int width, int WhiteCenter)
{
	vector<Point>cList;
	bool **visited;
	Mat img;
	resize(ori_img, img, Size(height, width), 0, 0, CV_INTER_CUBIC);      //구현해야될 부분 이미지 사이즈 조절해주는 함수
	height = img.rows; width = img.cols;   //영상 크기
	Mat img_gray(Size(width, height), 1);

	unsigned char**out;
	out = (unsigned char**)malloc(sizeof(unsigned char*)*height);
	for (int i = 0; i < height; i++) {
		out[i] = (unsigned char*)malloc(sizeof(unsigned char)*width);
	}
	cvtColor(img, img_gray, CV_BGR2GRAY);

	for (int h = 0; h < height; h++) {
		for (int w = 0; w < width; w++) {
			out[h][w] = img_gray.at<unsigned char>(h, w);
		}
	}
	harrisCorner(out, cList, th, height, width);      //특징점 추출 함수   (흑백영상, 특징점 저장할 벡터,임계값)
	sort(cList.begin(), cList.end(), cmp);   //저장된 특징점 정렬

	visited = (bool**)malloc(sizeof(bool*)*height);
	for (int i = 0; i < height; i++) {
		visited[i] = (bool*)malloc(sizeof(bool)*width);
		memset(visited[i], false, sizeof(bool)*width);
	}

	vector<vector<Point>> face = bfs(pointDist, keyPointCnt, cList);
	vector<Point>featurePoints = centerValue(face);
	Point finalPos = faceCenter(featurePoints);
	Point line = rectLength(finalPos, featurePoints);

	int color[3] = { 255,0,255 };
	pointPrint(finalPos.x, finalPos.y, color, img);
	Point pt1(finalPos.x - line.x, finalPos.y - line.x*1.3);
	Point pt2(finalPos.x + line.x, finalPos.y + line.x*1.3);

	/*
	   0,0 으로부터 가장 가까운 특징점에 중심값과 0,0으로부터 가장 짧은 거리에 있는 특징점 간에 중심값 구하기
	*/
	double x = sqrt(featurePoints[0].x*featurePoints[0].x + featurePoints[0].y*featurePoints[0].y);
	Point y = featurePoints[0];
	for (int i = 1; i < featurePoints.size(); i++)
	{
		if (x > sqrt((featurePoints[i].x*featurePoints[i].x) + (featurePoints[i].y * featurePoints[i].y)))
		{
			x = sqrt(featurePoints[i].x*featurePoints[i].x + featurePoints[i].y*featurePoints[i].y);
			y = featurePoints[i];
		}
	}

	double xx = sqrt(cList[0].x*cList[0].x + cList[0].y*cList[0].y);
	Point yy = cList[0];
	for (int i = 1; i < cList.size(); i++)
	{
		if (x > sqrt((cList[i].x*cList[i].x) + (cList[i].y * cList[i].y)))
		{
			xx = sqrt(cList[i].x*cList[i].x + cList[i].y*cList[i].y);
			yy = cList[i];
		}
	}

	Point finalPoint = Point((y.x + yy.x) / 2, (y.y + yy.y) / 2);
	//double PlusColSize = width - 2 * finalPoint.x;
	double PlusColSize = 2 * (WhiteCenter - finalPoint.x);
	double PlusRowSize = PlusColSize * (7.0 / 5.0) * ((double)ori_img.cols / (double)ori_img.rows);
	
	

	//특징점 초록색 점 찍기
	for (int i = 0; i < cList.size(); i++) {
		int px = cList[i].x;
		int py = cList[i].y;
		Vec3b RGB = img.at<Vec3b>(py, px);
		int color[3] = { 0,255,0 };
		pointPrint(px, py, color, img);
	}

	//특징점 파란색 점 찍기
	for (int i = 0; i < featurePoints.size(); i++) {
		int px = featurePoints[i].x;
		int py = featurePoints[i].y;
		Vec3b RGB = img.at<Vec3b>(py, px);
		int color[3] = { 255,0,0 };
		pointPrint(px, py, color, img);
	}



	rectangle(img, finalPoint, Point(finalPoint.x + PlusColSize, finalPoint.y + PlusRowSize), Scalar(255, 200, 100), 5);//사각형 그리기
	Mat faceDomain= faceDetected(img, finalPoint.x, finalPoint.y, finalPoint.x + PlusColSize, finalPoint.y + PlusRowSize);



	std::string OutName = std::to_string(count);
	
	return faceDomain;
}
Mat faceDetected(Mat img,int startX,int startY,int destX,int destY) {
	Mat faceDected(Size(destX-startX,destY-startY), CV_8UC1);
	Mat img_resize, graytmp;
	//resize(ori_img, tmpImg, Size(512, 512), 0, 0, CV_INTER_CUBIC);
	cvtColor(img, graytmp, CV_BGR2GRAY);
	for (int i = startY; i < destY; i++) {
		for (int j = startX; j <destX; j++) {
			faceDected.at<unsigned char>(i - startY, j - startX) = graytmp.at<unsigned char>(i, j);
		}
	}
	resize(faceDected, img_resize, Size(256, 256), 0, 0, CV_INTER_CUBIC);
	return img_resize;
}

void calculateHisogram(int hist[256], Mat img) {
	for (int h = 0; h < SIZE; h++) {
		for (int w = 0; w < SIZE; w++) {
			hist[img.at<uchar>(h, w)]++;
		}
	}
}


int  basicGlobalThresholing(int hist[256]) {
	int globalT = 128;
	int m1 = 0, m2 = 0;
	int down1 = 0, down2 = 0;
	for (int x = 0; x < 6; x++) {
		for (int i = 0; i < SIZE; i++) {
			if (i < globalT) {
				m1 += i * hist[i];
				down1 += hist[i];
			}
			else {
				m2 += i * hist[i];
				down2 += hist[i];
			}
		}
		globalT = ((m1 / down1) + (m2 / down2)) / 2;
	}
	return globalT;
}

void toBinaryImage(Mat ori, Mat dst, int globalT) {
	for (int h = 0; h < SIZE; h++) {
		for (int w = 0; w < SIZE; w++) {
			if (ori.at<uchar>(h, w) < globalT) {
				dst.at<uchar>(h, w) = 0;
			}
			else {
				dst.at<uchar>(h, w) = 255;
			}
		}
	}
}

void cutCircle(Mat img, int D) {
	int center = SIZE / 2;
	for (int h = 0; h < SIZE; h++) {
		for (int w = 0; w < SIZE; w++) {
			if ((h - center)*(h - center) + (w - center)*(w - center) > D * D) {
				img.at<uchar>(h, w) = 255;
			}
		}
	}
}


unsigned char** allocMem(int height, int width) {
	unsigned char** ret = new unsigned char*[height];
	for (int i = 0; i < height; i++) {
		ret[i] = new unsigned char[width];
		memset(ret[i], 0, sizeof(unsigned char)*width);
	}
	return ret;
}

void harrisCorner(unsigned char** Img_Ori, vector<Point>& corners, double th, int height, int width)
{

	int W = width;
	int H = height;
	unsigned char** Img = Img_Ori;

	double Ix; // Ix
	double Iy; // Iy

	double** dx2 = MemAlloc2D(H, W, 0);  //Ix의 제곱
	double** dy2 = MemAlloc2D(H, W, 0); //Iy의 제곱
	double** dxy = MemAlloc2D(H, W, 0); //Ix * Iy의 값

	double** gdx2 = MemAlloc2D(H, W, 0);//행렬 M의 원소들
	double** gdy2 = MemAlloc2D(H, W, 0);//M은 (gdx2 gdxy)
	double** gdxy = MemAlloc2D(H, W, 0);//    (gdxy gdy2) 로 이루어짐

	double tx2, ty2, txy; // M을 계산할 때 사용하는 임시 변수
	double** R = MemAlloc2D(H, W, 0); //R을 저장하는 것.
	double k = 0.04; // 코너응답함수의 상수값 00.4~ 0.06이 적절

	double Gaussian[5][5] = { 1, 4 , 6 , 4 , 1 ,
			   4, 16, 24, 16, 4,
			   6, 24, 36, 24, 6,
			   4, 16, 24, 16, 4 ,
			   1, 4 , 6 , 4 , 1 }; // W(람다x,람다y) => 가우시안분포값                  
	for (int h = 0; h < 5; h++)
		for (int w = 0; w < 5; w++)
		{
			Gaussian[h][w] /= 256;
		}

	//Ix와 Iy구하기 
	for (int h = 1; h < H - 1; h++)
		for (int w = 1; w < W - 1; w++)
		{
			Ix = (Img[h - 1][w + 1] + Img[h][w + 1] + Img[h + 1][w + 1] - Img[h - 1][w - 1] - Img[h][w - 1] - Img[h + 1][w - 1]) / 6.0;
			Iy = (Img[h + 1][w - 1] + Img[h + 1][w] + Img[h + 1][w + 1] - Img[h - 1][w - 1] - Img[h - 1][w] - Img[h - 1][w + 1]) / 6.0;
			dx2[h][w] = Ix * Ix;
			dy2[h][w] = Iy * Iy;
			dxy[h][w] = Ix * Iy;
		}

	//M 구하기
	for (int h = 2; h < H - 2; h++)
		for (int w = 2; w < W - 2; w++)
		{
			tx2 = ty2 = txy = 0;
			for (int hh = 0; hh < 5; hh++)
				for (int ww = 0; ww < 5; ww++)
				{
					tx2 += (dx2[h + hh - 2][w + hh - 2] * Gaussian[hh][ww]);
					ty2 += (dy2[h + hh - 2][w + ww - 2] * Gaussian[hh][ww]);
					txy += (dxy[h + hh - 2][w + ww - 2] * Gaussian[hh][ww]);
					//M을 계산하는 과정
				}

			gdx2[h][w] = tx2;
			gdy2[h][w] = ty2;
			gdxy[h][w] = txy;
		}

	//M을 이용하여 R값 계산
	for (int h = 2; h < H - 2; h++)
		for (int w = 2; w < W - 2; w++)
		{
			R[h][w] = (gdx2[h][w] * gdy2[h][w] - gdxy[h][w] * gdxy[h][w])
				- k * (gdx2[h][w] + gdy2[h][w])*(gdx2[h][w] + gdy2[h][w]);
		}

	//R값을 주변 8개와 비교
	for (int h = 2; h < H - 2; h++)
		for (int w = 2; w < W - 2; w++)
		{
			if (R[h][w] > th)
			{
				if (R[h][w] > R[h - 1][w] && R[h][w] > R[h - 1][w + 1] &&
					R[h][w] > R[h][w + 1] && R[h][w] > R[h + 1][w + 1] &&
					R[h][w] > R[h + 1][w] && R[h][w] > R[h + 1][w - 1] &&
					R[h][w] > R[h][w - 1] && R[h][w] > R[h - 1][w - 1])
				{
					corners.push_back(Point(w, h));
				}
			}
		}
}

double** MemAlloc2D(int nHeight, int nWidth, unsigned char nInitVal)
{
	double** rtn = new double*[nHeight];
	for (int n = 0; n < nHeight; n++)
	{
		rtn[n] = new double[nWidth];
		memset(rtn[n], nInitVal, sizeof(double)* nWidth);

	}

	return rtn;
}

bool cmp(Point a, Point b) {
	if (a.y < b.y)
		if (a.x < b.y)
			return true;
		else
			return false;
	else
		return false;
}

Point rectLength(Point a, vector<Point> b)
{
	int xmax = abs(a.x - b[0].x);
	int ymax = abs(a.y - b[0].y);

	for (int i = 0; i < b.size(); i++)
	{
		if (abs(a.x - b[i].x) > xmax)
			xmax = abs(a.x - b[i].x);
		if (abs(a.y - b[i].y) > ymax)
			ymax = abs(a.y - b[i].y);
	}

	return Point(xmax, ymax);
}

void pointPrint(int x, int y, int color[3], Mat img) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			img.at<Vec3b>(y + dy[i], x + dx[j]) = Vec3b(color[0], color[1], color[2]);
		}
	}
}

vector<vector<Point>> bfs(int pointDist, int keyPointCnt, vector<Point> cList) {
	vector<vector<Point>>face;
	bool **visited;
	visited = (bool**)malloc(sizeof(bool*)*height);
	for (int i = 0; i < height; i++) {
		visited[i] = (bool*)malloc(sizeof(bool)*width);
		memset(visited[i], false, sizeof(bool)*width);
	}
	for (int i = 0; i < cList.size(); i++) {
		int currentPosX = cList[i].x;
		int currentPosY = cList[i].y;
		vector<Point> newV;
		queue<Point>q;
		q.push(Point(currentPosX, currentPosY));
		while (!q.empty()) {
			currentPosX = q.front().x;
			currentPosY = q.front().y;
			q.pop();
			if (!visited[currentPosY][currentPosX]) {
				visited[currentPosY][currentPosX] = true;
				newV.push_back(Point(currentPosX, currentPosY));
				for (int j = i + 1; j < cList.size(); j++) {
					int nextPosX = cList[j].x;
					int nextPosY = cList[j].y;
					int diffX = abs(currentPosX - nextPosX);
					int diffY = abs(currentPosY - nextPosY);
					if (!visited[nextPosY][nextPosX] && diffX <= pointDist && diffY <= pointDist) {
						q.push(Point(nextPosX, nextPosY));
					}
				}
			}
		}
		if (newV.size() >= keyPointCnt) {
			face.push_back(newV);
		}
	}
	if (face.size() < 5) {
		int specPoint = keyPointCnt - 1;
		face = bfs(pointDist, specPoint, cList);
	}

	return face;
}

vector<Point> centerValue(vector<vector<Point>>& c)
{
	int xpoint;
	int ypoint;
	vector<Point> rtn;


	for (int i = 0; i < c.size(); i++)
	{
		vector<Point> a = c[i];
		double Xsum = 0.0;
		double Ysum = 0.0;


		for (int i = 0; i < a.size(); i++)
			Xsum += a[i].x;
		for (int j = 0; j < a.size(); j++)
			Ysum += a[j].y;



		xpoint = (double)Xsum / (double)(a.size());
		ypoint = (double)Ysum / (double)(a.size());

		rtn.push_back(Point((int)xpoint, (int)ypoint));
	}

	return rtn;
}

Point faceCenter(vector<Point> a)
{
	double xpoint;
	double ypoint;
	double Xsum = 0;
	double Ysum = 0;

	for (int i = 0; i < a.size(); i++)
		Xsum += a[i].x;
	for (int j = 0; j < a.size(); j++)
		Ysum += a[j].y;

	xpoint = (double)Xsum / (double)(a.size());
	ypoint = (double)Ysum / (double)(a.size());

	return Point((int)xpoint, (int)ypoint);

}

unsigned char BicubicInterpolation(unsigned char** In, int nHegiht_Ori, int nWidth_Ori, double h_Cvt, double w_Cvt, int size)
{

	int h = (int)h_Cvt;
	int w = (int)w_Cvt;

	if (h > 1 && h < nHegiht_Ori - 2 && w > 1 && w < nWidth_Ori - 2)
	{
		double n = w_Cvt - w;
		double nn = h_Cvt - h;
		double a = (double)1 / (double)6;
		double c = (double)1 / (double)2;
		double b = (double)1 / (double)3;


		double x1 = (double)pow(n, 3) * (-a * In[h - 1][w - 1] + c * In[h - 1][w] - c * In[h - 1][w + 1] + a * In[h - 1][w + 2])
			+ (double)pow(n, 2) * (c * In[h - 1][w - 1] - In[h - 1][w] + c * In[h - 1][w + 1])
			+ (double)n * (-b * In[h - 1][w - 1] - c * In[h - 1][w] + In[h - 1][w + 1] - a * In[h - 1][w + 2])
			+ (In[h - 1][w]);
		double x2 = (double)pow(n, 3) * (-a * In[h][w - 1] + c * In[h][w] - c * In[h][w + 1] + a * In[h][w + 2])
			+ (double)pow(n, 2) * (c * In[h][w - 1] - In[h][w] + c * In[h][w + 1])
			+ (double)n * (-b * In[h][w - 1] - c * In[h][w] + In[h][w + 1] - a * In[h][w + 2])
			+ (In[h][w]);
		double x3 = (double)pow(n, 3) * (-a * In[h + 1][w - 1] + c * In[h + 1][w] - c * In[h + 1][w + 1] + a * In[h + 1][w + 2])
			+ (double)pow(n, 2) * (c * In[h + 1][w - 1] - In[h + 1][w] + c * In[h + 1][w + 1])
			+ (double)n * (-b * In[h + 1][w - 1] - c * In[h + 1][w] + In[h + 1][w + 1] - a * In[h + 1][w + 2])
			+ (In[h + 1][w]);
		double x4 = (double)pow(n, 3) * (-a * In[h + 2][w - 1] + c * In[h + 2][w] - c * In[h + 2][w + 1] + a * In[h + 2][w + 2])
			+ (double)pow(n, 2) * (c * In[h + 2][w - 1] - In[h + 2][w] + c * In[h + 2][w + 1])
			+ (double)n * (-b * In[h + 2][w - 1] - c * In[h + 2][w] + In[h + 2][w + 1] - a * In[h + 2][w + 2])
			+ (In[h + 2][w]);

		double answer = (double)pow(nn, 3) * (-a * x2 + c * x2 - c * x3 + a * x4)
			+ (double)pow(nn, 2) * (c * x1 - x2 + c * x3)
			+ (double)nn * (-b * x1 - c * x2 + x3 - a * x4)
			+ (x2);

		return answer;
	}
	else
	{
		return In[h][w];
	}
}

vector<Point> abc(Point a, unsigned char** Gray_Img, int height, int width, int th, int rowsize, int colsize)
{
	vector<Point> rtn;

	int rect1, rect2, rect3;
	rect1 = rect2 = rect3 = 0;
	for (int h = a.y; h < a.y + height; h++)
		for (int w = a.x; w < a.x + width; w++)
		{
			rect1 = rect2 = rect3 = 0;
			for (int hh = h; hh < h + (rowsize / 3); hh++)
				for (int ww = w; ww < w + colsize; ww++)
					rect1 += Gray_Img[hh][ww];
			for (int hh = h + (rowsize / 3); hh < h + 2 * (rowsize / 3); hh++)
				for (int ww = w; ww < w + colsize; ww++)
					rect2 += Gray_Img[hh][ww];
			for (int hh = h + 2 * (rowsize / 3); hh < h + 3 * (rowsize / 3); hh++)
				for (int ww = w; ww < w + colsize; ww++)
					rect3 += Gray_Img[hh][ww];

			if (rect2 - rect1 > th && rect2 - rect3 > th)
				rtn.push_back(Point(w, h));

		}

	return rtn;
}

IplImage* convertImageRGBtoHSV(const IplImage *imageRGB)
{
	float fR, fG, fB;
	float fH, fS, fV;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

	// Create a blank HSV image
	IplImage *imageHSV = cvCreateImage(cvGetSize(imageRGB), 8, 3);
	if (!imageHSV || imageRGB->depth != 8 || imageRGB->nChannels != 3) {
		printf("ERROR in convertImageRGBtoHSV()! Bad input image.\n");
		exit(1);
	}

	int h = imageRGB->height;      // Pixel height.
	int w = imageRGB->width;      // Pixel width.
	int rowSizeRGB = imageRGB->widthStep;   // Size of row in bytes, including extra padding.
	char *imRGB = imageRGB->imageData;   // Pointer to the start of the image pixels.
	int rowSizeHSV = imageHSV->widthStep;   // Size of row in bytes, including extra padding.
	char *imHSV = imageHSV->imageData;   // Pointer to the start of the image pixels.
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			// Get the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			uchar *pRGB = (uchar*)(imRGB + y * rowSizeRGB + x * 3);
			int bB = *(uchar*)(pRGB + 0);   // Blue component
			int bG = *(uchar*)(pRGB + 1);   // Green component
			int bR = *(uchar*)(pRGB + 2);   // Red component

			// Convert from 8-bit integers to floats.
			fR = bR * BYTE_TO_FLOAT;
			fG = bG * BYTE_TO_FLOAT;
			fB = bB * BYTE_TO_FLOAT;

			// Convert from RGB to HSV, using float ranges 0.0 to 1.0.
			float fDelta;
			float fMin, fMax;
			int iMax;
			// Get the min and max, but use integer comparisons for slight speedup.
			if (bB < bG) {
				if (bB < bR) {
					fMin = fB;
					if (bR > bG) {
						iMax = bR;
						fMax = fR;
					}
					else {
						iMax = bG;
						fMax = fG;
					}
				}
				else {
					fMin = fR;
					fMax = fG;
					iMax = bG;
				}
			}
			else {
				if (bG < bR) {
					fMin = fG;
					if (bB > bR) {
						fMax = fB;
						iMax = bB;
					}
					else {
						fMax = fR;
						iMax = bR;
					}
				}
				else {
					fMin = fR;
					fMax = fB;
					iMax = bB;
				}
			}
			fDelta = fMax - fMin;
			fV = fMax;            // Value (Brightness).
			if (iMax != 0) {         // Make sure it's not pure black.
				fS = fDelta / fMax;      // Saturation.
				float ANGLE_TO_UNIT = 1.0f / (6.0f * fDelta);   // Make the Hues between 0.0 to 1.0 instead of 6.0
				if (iMax == bR) {      // between yellow and magenta.
					fH = (fG - fB) * ANGLE_TO_UNIT;
				}
				else if (iMax == bG) {      // between cyan and yellow.
					fH = (2.0f / 6.0f) + (fB - fR) * ANGLE_TO_UNIT;
				}
				else {            // between magenta and cyan.
					fH = (4.0f / 6.0f) + (fR - fG) * ANGLE_TO_UNIT;
				}
				// Wrap outlier Hues around the circle.
				if (fH < 0.0f)
					fH += 1.0f;
				if (fH >= 1.0f)
					fH -= 1.0f;
			}
			else {
				// color is pure Black.
				fS = 0;
				fH = 0;   // undefined hue
			}

			// Convert from floats to 8-bit integers.
			int bH = (int)(0.5f + fH * 255.0f);
			int bS = (int)(0.5f + fS * 255.0f);
			int bV = (int)(0.5f + fV * 255.0f);

			// Clip the values to make sure it fits within the 8bits.
			if (bH > 255)
				bH = 255;
			if (bH < 0)
				bH = 0;
			if (bS > 255)
				bS = 255;
			if (bS < 0)
				bS = 0;
			if (bV > 255)
				bV = 255;
			if (bV < 0)
				bV = 0;

			// Set the HSV pixel components.
			uchar *pHSV = (uchar*)(imHSV + y * rowSizeHSV + x * 3);
			*(pHSV + 0) = bH;      // H component
			*(pHSV + 1) = bS;      // S component
			*(pHSV + 2) = bV;      // V component
		}
	}
	return imageHSV;
}

IplImage* convertImageHSVtoRGB(const IplImage *imageHSV)
{
	float fH, fS, fV;
	float fR, fG, fB;
	const float FLOAT_TO_BYTE = 255.0f;
	const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;

	// Create a blank RGB image
	IplImage *imageRGB = cvCreateImage(cvGetSize(imageHSV), 8, 3);
	if (!imageRGB || imageHSV->depth != 8 || imageHSV->nChannels != 3) {
		printf("ERROR in convertImageHSVtoRGB()! Bad input image.\n");
		exit(1);
	}

	int h = imageHSV->height;         // Pixel height.
	int w = imageHSV->width;         // Pixel width.
	int rowSizeHSV = imageHSV->widthStep;      // Size of row in bytes, including extra padding.
	char *imHSV = imageHSV->imageData;      // Pointer to the start of the image pixels.
	int rowSizeRGB = imageRGB->widthStep;      // Size of row in bytes, including extra padding.
	char *imRGB = imageRGB->imageData;      // Pointer to the start of the image pixels.
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			// Get the HSV pixel components
			uchar *pHSV = (uchar*)(imHSV + y * rowSizeHSV + x * 3);
			int bH = *(uchar*)(pHSV + 0);   // H component
			int bS = *(uchar*)(pHSV + 1);   // S component
			int bV = *(uchar*)(pHSV + 2);   // V component

			// Convert from 8-bit integers to floats
			fH = (float)bH * BYTE_TO_FLOAT;
			fS = (float)bS * BYTE_TO_FLOAT;
			fV = (float)bV * BYTE_TO_FLOAT;

			// Convert from HSV to RGB, using float ranges 0.0 to 1.0
			int iI;
			float fI, fF, p, q, t;

			if (bS == 0) {
				// achromatic (grey)
				fR = fG = fB = fV;
			}
			else {
				// If Hue == 1.0, then wrap it around the circle to 0.0
				if (fH >= 1.0f)
					fH = 0.0f;

				fH *= 6.0;         // sector 0 to 5
				fI = floor(fH);      // integer part of h (0,1,2,3,4,5 or 6)
				iI = (int)fH;         //      "      "      "      "
				fF = fH - fI;         // factorial part of h (0 to 1)

				p = fV * (1.0f - fS);
				q = fV * (1.0f - fS * fF);
				t = fV * (1.0f - fS * (1.0f - fF));

				switch (iI) {
				case 0:
					fR = fV;
					fG = t;
					fB = p;
					break;
				case 1:
					fR = q;
					fG = fV;
					fB = p;
					break;
				case 2:
					fR = p;
					fG = fV;
					fB = t;
					break;
				case 3:
					fR = p;
					fG = q;
					fB = fV;
					break;
				case 4:
					fR = t;
					fG = p;
					fB = fV;
					break;
				default:      // case 5 (or 6):
					fR = fV;
					fG = p;
					fB = q;
					break;
				}
			}

			// Convert from floats to 8-bit integers
			int bR = (int)(fR * FLOAT_TO_BYTE);
			int bG = (int)(fG * FLOAT_TO_BYTE);
			int bB = (int)(fB * FLOAT_TO_BYTE);

			// Clip the values to make sure it fits within the 8bits.
			if (bR > 255)
				bR = 255;
			if (bR < 0)
				bR = 0;
			if (bG > 255)
				bG = 255;
			if (bG < 0)
				bG = 0;
			if (bB > 255)
				bB = 255;
			if (bB < 0)
				bB = 0;

			// Set the RGB pixel components. NOTE that OpenCV stores RGB pixels in B,G,R order.
			uchar *pRGB = (uchar*)(imRGB + y * rowSizeRGB + x * 3);
			*(pRGB + 0) = bB;      // B component
			*(pRGB + 1) = bG;      // G component
			*(pRGB + 2) = bR;      // R component
		}
	}
	return imageRGB;
}