#include<iostream>
#include<opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;



//��������
//int removeFrequnce();
//int dftDemo();
//int ifftDemo();
//int mouseROI();


void on_mouse(int EVENT, int x, int y, int flags, void* userdata);
int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat);
//int calcVisibalMag(cv::Mat srcMat, cv::Mat & dstMat);
//int calcVisbalDft(cv::Mat srcMat, cv::Mat & magMat, cv::Mat & ph, double & normVal);
//int calcDft2Image(cv::Mat magMat, cv::Mat ph, double normVal, cv::Mat &dstMat);

std::vector<Point>  mousePoints;
Point points;


/***************************************�����Ӧ����*******************************************/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{

	Mat hh;
	hh = *(Mat*)userdata;
	Point p(x, y);
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
	{
		points.x = x;
		points.y = y;
		mousePoints.push_back(points);
		circle(hh, points, 4, cvScalar(255, 255, 255), -1);
		imshow("mouseCallback", hh);
	}
	break;
	}

}

//�õ�ֻ�е�Ƶ���ֵ�ͼ
int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat)
{

	vector<vector<Point>> contours;
	cv::Mat selectMat;

	cv::Mat m = cv::Mat::zeros(srcMat.size(), CV_32F);

	m = 1;

	if (!srcMat.empty()) {
		srcMat.copyTo(selectMat);
		srcMat.copyTo(dstMat);
	}
	else {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	namedWindow("mouseCallback");
	imshow("mouseCallback", selectMat);
	setMouseCallback("mouseCallback", on_mouse, &selectMat);
	waitKey(0);
	destroyAllWindows();
	//����roi
	contours.push_back(mousePoints);
	if (contours[0].size() < 3) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	drawContours(m, contours, 0, Scalar(0), -1);  //������Ƶ����

	m.copyTo(dstMat);

	return 0;
}

//�õ�ֻ�и�Ƶ���ֵ�ͼ
int selectPolygon2(cv::Mat srcMat, cv::Mat &dstMat)
{

	vector<vector<Point>> contours;
	cv::Mat selectMat;

	cv::Mat m = cv::Mat::zeros(srcMat.size(), CV_32F);

	m = 1;

	if (!srcMat.empty()) {
		srcMat.copyTo(selectMat);
		srcMat.copyTo(dstMat);
	}
	else {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	namedWindow("mouseCallback");
	imshow("mouseCallback", selectMat);
	setMouseCallback("mouseCallback", on_mouse, &selectMat);
	waitKey(0);
	destroyAllWindows();
	//����roi
	contours.push_back(mousePoints);
	if (contours[0].size() < 3) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	drawContours(m, contours, 0, Scalar(0), -1);

	//����1��ͼ��ȥ0���͵õ�ֻ�и�Ƶ���ֵ�
	Mat m1;
	m1.create(srcMat.size(), srcMat.type());//����һ��ָ����С��ָ�����͵ľ�����
	m1 = Scalar(1);

	//dstMat = m1 - m;//��Ƶ����Ϊ1����Ƶ����Ϊ0��������Ƶ����
	float beta, alpha;
	alpha = 1;
	beta = -1;
	addWeighted(m1, alpha, m, beta, 0.0, dstMat);


	

	//m.copyTo(dstMat);

	return 0;
}



//�Ӹ���־t���ֱ�������Ƶ�ź��뱣����Ƶ�źŵ�ͼƬ,t=1ʱ���µ�Ƶ�ģ�Ĭ�ϣ�,t=0ʱ���¸�Ƶ��

int ifftDemo(cv::Mat src, cv::Mat &dst,int t=1) 
{


	int m = getOptimalDFTSize(src.rows); //2,3,5�ı����и���Ч�ʵĸ���Ҷ�任
	int n = getOptimalDFTSize(src.cols);
	Mat padded;
	//�ѻҶ�ͼ��������Ͻ�,���ұߺ��±���չͼ��,��չ�������Ϊ0;
	copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));
	//planes[0]Ϊdft�任��ʵ����planes[1]Ϊ�鲿��phΪ��λ�� plane_true=magΪ��ֵ
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat planes_true = Mat_<float>(padded);
	Mat ph = Mat_<float>(padded);
	Mat complexImg;
	//��ͨ��complexImg����ʵ�������鲿
	merge(planes, 2, complexImg);
	//���ϱߺϳɵ�mat���и���Ҷ�任,***֧��ԭ�ز���***,����Ҷ�任���Ϊ����.ͨ��1�����ʵ��,ͨ����������鲿
	dft(complexImg, complexImg);
	//�ѱ任��Ľ���ָ����mat,һ��ʵ��,һ���鲿,�����������
	split(complexImg, planes);

	//---------------�˲���Ŀ��Ϊ���õ���ʾ��ֵ---�����ָ�ԭͼʱ�����ٴ���һ��-------------------------
	magnitude(planes[0], planes[1], planes_true);//������mag
	phase(planes[0], planes[1], ph);//��λ��ph
	Mat A = planes[0];
	Mat B = planes[1];
	Mat mag = planes_true;

	mag += Scalar::all(1);//�Է�ֵ��1
	//������ķ�ֵһ��ܴ󣬴ﵽ10^4,ͨ��û�а취��ͼ������ʾ��������Ҫ�������log��⡣
	log(mag, mag);

	//ȡ�����е����ֵ�����ں�����ԭʱȥ��һ��
	double maxVal;
	minMaxLoc(mag, 0, &maxVal, 0, 0);

	//�޼�Ƶ��,���ͼ����л������������Ļ�,����Ƶ���ǲ��ԳƵ�,���Ҫ�޼�
	mag = mag(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	ph = ph(Rect(0, 0, mag.cols & -2, mag.rows & -2));
	Mat _magI = mag.clone();
	//�����ȹ�һ��������ʾ��Χ��
	normalize(_magI, _magI, 0, 1, CV_MINMAX);
	//imshow("before rearrange", _magI);

	//��ʾ����Ƶ��ͼ
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;

	//������������Ϊ��׼����magͼ��ֳ��Ĳ���
	Mat tmp;
	Mat q0(mag, Rect(0, 0, cx, cy));
	Mat q1(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(mag, mag, 0, 1, CV_MINMAX);
	//imshow("ԭͼ�Ҷ�ͼ", src);
	//imshow("Ƶ�׷���", mag);
	mag = mag * 255;
	imwrite("ԭƵ��.jpg", mag);
	/*--------------------------------------------------*/

	mag = mag / 255;
	cv::Mat mask;
	Mat proceMag;

	//�������¸�Ƶ��Ϣ���������µ�Ƶ��Ϣ
	if (t=1)
	{
		selectPolygon(mag, mask);
	}
	else
	{
		selectPolygon2(mag, mask);
	}

	mag = mag.mul(mask);

	proceMag = mag * 255;
	imwrite("�����Ƶ��.jpg", proceMag);

	//ǰ�����跴����һ�飬Ŀ����Ϊ����任��ԭͼ
	Mat q00(mag, Rect(0, 0, cx, cy));
	Mat q10(mag, Rect(cx, 0, cx, cy));
	Mat q20(mag, Rect(0, cy, cx, cy));
	Mat q30(mag, Rect(cx, cy, cx, cy));

	//��������
	q00.copyTo(tmp);
	q30.copyTo(q00);
	tmp.copyTo(q30);
	q10.copyTo(tmp);
	q20.copyTo(q10);
	tmp.copyTo(q20);

	mag = mag * maxVal;//����һ���ľ���ԭ 
	exp(mag, mag);//��Ӧ��ǰ��ȥ����
	mag = mag - Scalar::all(1);//��Ӧǰ��+1
	polarToCart(mag, ph, planes[0], planes[1]);//�ɷ�����mag����λ��ph�ָ�ʵ��planes[0]���鲿planes[1]
	merge(planes, 2, complexImg);//��ʵ���鲿�ϲ�


	//-----------------------����Ҷ����任-----------------------------------
	Mat ifft(Size(src.cols, src.rows), CV_8UC1);
	//����Ҷ��任
	idft(complexImg, ifft, DFT_REAL_OUTPUT);
	normalize(ifft, ifft, 0, 1, CV_MINMAX);

	Rect rect(0, 0, src.cols, src.rows);
	dst = ifft(rect);
	dst = dst * 255;

	cv::Mat dspMat;
	dst.convertTo(dspMat, CV_8UC1);
	imshow("dst", dspMat);
	imshow("src", src);
	/*waitKey(0);*/

	return 0;

}


int main() {

	Mat src1 = imread("��.jpg", 0);
	Mat src2 = imread("��2.jpg",0);

	Mat dst1,dst2;


	ifftDemo(src1, dst1,1); //������ĵ�Ƶ��Ϣ���ⲿ����
	ifftDemo(src2, dst2, 2); //�����ǵĸ�Ƶ��Ϣ

	Mat imgAdd;

	//imgAdd = dst1 + dst2;
	float beta, alpha;
	alpha = 0.5;
	beta = (1.0 - alpha);
	addWeighted(src1, alpha, src2, beta, 0.0, imgAdd);
	

	imshow("ͼƬ���", imgAdd);

	waitKey(0);


}