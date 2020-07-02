#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <math.h>
#include <iostream>
#define ABC 1
using namespace dlib;
using namespace std;


typedef struct VIDEODATA
{
	unsigned char head[2];			 	//报文头，固定格式为0xaaaa
	unsigned char payloadsize[2];		//真实数据的长度，小于64k
	unsigned char flag;					//报文标识，固定为0x58
	unsigned char type;					//数据类型：1=盹睡视频；2=行为视频
	unsigned char starttime[4];			//开始时间，盹睡时间
	unsigned char endtime[4];			//上报时间
	unsigned char total;				//总共图片张数
	unsigned char index;				//当前图片索引
	unsigned char payload[1024*63];		//真实数据（图片数据）
	unsigned char check;				//校验和
}VIDEODATA;
/*
void GenerateArray(int x_len, int y_len) {//x_len，y_len是帧数。
	int i_start = 0;
	while (i_start<x_len) {
		int j_start = 0;
		DT m = 0.0;
		for (; j_start <y_len; j_start++) {
			int k_start = 0;
			for (; k_start < 12; k_start++) {
				m += pow(X[i_start][k_start] - Y[j_start][k_start], 2);
			}
			dataArray[i_start][j_start] = m;//将欧氏距离矩阵存放到数组中
			m = 0;//clear zero
		}
		i_start++;
	}
	printDataArray(dataArray, x_len, y_len);
}*/
static cv::Mat images(cv::VideoCapture cam){
	

	cv::VideoWriter writer;//写入视频对象
	//输出文件名
	std::string outputFile = "output.mp4";
	
	//cam.open("rtsp://admin:qwedcvfr@192.168.1.198/cam/realmonitor?channel=1&subtype=0");
	//cam.open(0);
	if (!cam.isOpened()) {
		cout << "open error!" << endl;
	}

	//writer.open(outputFile,CV_FOURCC("M","J","P","G"),25,cv::Size(320,240));

	cv::Mat m;
	cam.read(m);
	//cv::namedWindow("v");
	return m;
}

int test(int argc, char** argv)
{
	cv::VideoCapture cam;
	//int codec = cv::CV_FOURCC('M', 'P', '4', 'V');
	double fps_write = 7.0;
	cv::Size size_write = cv::Size(320, 240);
	cv::VideoWriter outputVideo;
	
	//outputVideo.open("sd.mp4", codec, fps_write, size_write, false);

	cam.open(0);
	cv::Mat s;
	for (;;) {
		s=images(cam);
		//outputVideo << s;

		cv::imshow("v", s);
		cv::waitKey(1000);
	}
	//outputVideo.release();

	getchar();
	return 0;
}
int main()
{
	/*
	unsigned char head[2];			 	//报文头，固定格式为0xaaaa
	unsigned char payloadsize[2];		//真实数据的长度，小于64k
	unsigned char flag;					//报文标识，固定为0x58
	unsigned char type;					//数据类型：1=盹睡视频；2=行为视频
	unsigned char starttime[4];			//开始时间，盹睡时间
	unsigned char endtime[4];			//上报时间
	unsigned char total;				//总共图片张数
	unsigned char index;				//当前图片索引
	unsigned char payload;				//真实数据（图片数据）
	unsigned char check;				//校验和
	*/
	VIDEODATA videodata;
	videodata.head[0] = 0xaa;
	videodata.head[1] = 0xaa;
	videodata.flag = 0x58;
	videodata.type = 1;
	std::vector<uchar> date_encode;
	FILE* fp;

	//三挡10间隔1秒 15间隔0.5秒 20秒长鸣 分层蜂鸣器判断
	
	float d, b, c, e, f;
	time_t start = time(NULL);
	cout << "开始时间start" << start << endl;
	char strTime[100], time_tp[10];
	string image_filename;
	string save_image_path = "image/";//保存截图路径

	float eyesleep = 0.23;
	int COUNTER = 0;
	int EYE_ASPECT_RATIO_THRESHOLD = 150;
	string video_save_time;
	cout << "---" << 338 << endl;
	b = sqrt(pow(339 - 340, 2) + pow(338 - 344, 2));
	printf("%f", b);
	cout << sqrt(3) << endl;
	clock_t  mid;
	try
	{

		//cv::VideoCapture cap("F:/数据/nmdate/SP197001010433.mp4"); //Video path
		//cv::VideoCapture cap("rtsp://admin:qwedcvfr@192.168.1.198/cam/realmonitor?channel=1&subtype=1");
		//cv::VideoCapture cap("rtsp://192.168.1.251:554/live/0/MAIN");
		cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "无法连接到相机" << endl;
			return 1;
		}
		image_window win;
		// 加载面检测和姿态估计模型。人脸分类器
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		//获取人脸检测器
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		/*
		if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {

			perror("socket");
			exit(1);
		}
		*/

		//抓取并处理帧，直到用户关闭主窗口。
		while (!win.is_closed())
		{
			int tp_count = 0;
			// Grab a frame
			cv::Mat temp;
			if (!cap.read(temp))
			{
				break;
			}
			//把OpenCV的Mat变成dlib可以处理的东西。注意这只是
			//包装Mat对象，它不复制任何东西。所以cimg只在
			//只要temp是有效的。也不要对温度做任何会导致它的事情
			//重新分配存储图像的内存，使其成为cimg
			//包含悬空指针。这基本上意味着您不应该修改temp
			//在使用cimg时。
			cv_image<bgr_pixel> cimg(temp);
			array2d<unsigned char> left_roi(32, 32);
			cout << "---=-=-=-=" << temp.rows << "-=-=" << temp.cols << endl;//图像的宽高  480  640
			// 检测面
			std::vector<rectangle> faces = detector(cimg);
			//找出每一张脸的姿势
			//cout << sizeof(faces) << endl;
			cout << faces.size() << endl;
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));
			cout << shapes.empty() << endl;
			if (!shapes.empty()) {
				double scale = 2.;
				int LEFT_EYE = 42;
				double center_x = 0.5 * (shapes[0].part(LEFT_EYE + 3).x() + shapes[0].part(LEFT_EYE).x());
				double center_y = 0.5 * (shapes[0].part(LEFT_EYE + 5).y() + shapes[0].part(LEFT_EYE + 1).y());
				double x_scale = scale * 0.5 * (shapes[0].part(LEFT_EYE + 3).x() - shapes[0].part(LEFT_EYE).x());
				double y_scale = scale * 0.5 * (shapes[0].part(LEFT_EYE + 5).y() - shapes[0].part(LEFT_EYE + 1).y());
				cout << center_x << "__" << center_y << "__" << x_scale << "__" << y_scale << "__" << endl;
				std::array<dpoint, 4> LEFT_ROI;

				LEFT_ROI[0](0) = center_x - x_scale;
				LEFT_ROI[0](1) = center_y + y_scale;
				//the topleft point
				LEFT_ROI[1](0) = LEFT_ROI[0](0);
				LEFT_ROI[1](1) = center_y - y_scale;
				//the topright point
				LEFT_ROI[2](0) = center_x + x_scale;
				LEFT_ROI[2](1) = LEFT_ROI[1](1);
				//the bottomright point
				LEFT_ROI[3](0) = LEFT_ROI[2](0);
				LEFT_ROI[3](1) = LEFT_ROI[0](1);

#if ABC
				cv::Mat cv_left_roi = toMat(left_roi);
				cv::namedWindow("left_roi");
				cv::imshow("left_roi", cv_left_roi);
				cv::waitKey(20);

#endif
				cout << "LEFT_ROI[0]" << LEFT_ROI[0] << endl; cout << "LEFT_ROI[1]" << LEFT_ROI[1] << "LEFT_ROI[2]" << LEFT_ROI[2] << endl;


			}

			if (!shapes.empty())
				for (int i = 42; i < 48; i++) {//左眼36-41   右眼42-47
					stringstream ss;
					ss << 1;
					string str = ss.str();
					
					int image_size = temp.cols * temp.rows;
					unsigned char* imageData = new unsigned char[image_size];

					//circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()), 3, cv::Scalar(0, 0, 255), -1);
					//cv::putText(temp,str, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()),cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1,cv::Scalar(0,0,255), 1, 1);
					cv::putText(temp, str, cvPoint(shapes[0].part(42).x(), shapes[0].part(42).y()), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1);
					cout << i << "-----" << shapes[0].part(i).x() << endl;
					cout << shapes[0].part(i) << endl;
					c = sqrt(pow(shapes[0].part(43).x() - shapes[0].part(47).x(), 2) + pow(shapes[0].part(43).y() - shapes[0].part(47).y(), 2));
					d = sqrt(pow(shapes[0].part(44).x() - shapes[0].part(46).x(), 2) + pow(shapes[0].part(44).y() - shapes[0].part(46).y(), 2));
					e = sqrt(pow(shapes[0].part(42).x() - shapes[0].part(45).x(), 2) + pow(shapes[0].part(42).y() - shapes[0].part(45).y(), 2));
					printf("c:%f--d:%f--e:%f", c, d, e);
					f = (c + d) / (2 * e);
					printf("--------%f", f);
					if (f < eyesleep) {
						COUNTER++;
						if (COUNTER >= EYE_ASPECT_RATIO_THRESHOLD) {

							

							tp_count++;
							cv::putText(temp, "You are Drowsy", cvPoint(shapes[0].part(42).x(), shapes[0].part(42).y()), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1);//左眼闭眼提示
							start = time(NULL);
							
							
							videodata.starttime[3] = (start & 0xff);
							videodata.starttime[2] = (start >> 8) & 0xff;
							videodata.starttime[1] = (start >> 16) & 0xff;
							videodata.starttime[0] = (start >> 24) & 0xff;

							
							strftime(strTime, sizeof(strTime), "%Y%m%d%H%M%S", localtime(&start));
							strftime(time_tp, sizeof(time_tp), "%S", localtime(&start));
						    video_save_time = strTime;
							image_filename = save_image_path + "JT"+strTime + to_string(tp_count) + ".JPG";
							//cv::putText(cv_img, "You are Drowsy", cv::Point(20, 20), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1, 1);
							cv::imwrite(image_filename, temp);
							/*int a = 0;
							for (int i = 0; i < temp.rows; i++)
							{
								for (int j = 0; j < temp.cols; j++)
								{
									imageData[a] = temp.at<uchar>(i, j);
									a++;
								}
							}
							videodata.payloadsize[0] = imageData;
							*/

							
							cv::imencode(".jpg", temp,date_encode);
							std::string str_encode(date_encode.begin(), date_encode.end());
							memcpy(videodata.payload, str_encode.c_str(), str_encode.size());
							videodata.index = tp_count;//当前图片索引
							videodata.check = atoi(strTime);
							//
							videodata.payloadsize[1] |= str_encode.size() & 0xff;
							videodata.payloadsize[0] |= (str_encode.size() >> 8) & 0xff;
							
							/*fp=fopen("123456","wb");
							fwrite(videodata.payload,1,str_encode.size(),fp);
							fclose(fp);
							*/
							//videodata.starttime[0] = strTime;
							//unsigned char* encode_data = new unsigned char[lSize];
							//for (int i = 0; i < lSize; i++) {
							//	encode_data[i] = img_encode[i];
							//}
							
						}

					}
					else
					{

						start = time(NULL);
						videodata.endtime[3] = (start & 0xff);
						videodata.endtime[2] = (start >> 8) & 0xff;
						videodata.endtime[1] = (start >> 16) & 0xff;
						videodata.endtime[0] = (start >> 24) & 0xff; //结束时间
						videodata.total = tp_count;//图片张数
						COUNTER = 0;
						tp_count = 0;
					}


					//shapes.push_back(pose_model(cimg,);

				}

			//全部显示在屏幕上dlib显示
#if ABC
			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
#endif

			//opencv显现
			//Mat fread
			//cv::imshow("image", cimg);
			//if(cv::waitKey(30)>=0):
		}
	}
	catch (serialization_error & e)
	{
		cout << "You need  model file to run this example." << endl;
	}
	catch (exception & e)
	{
		cout << e.what() << endl;
	}
}





//*************************opencv读取摄像头*****************************
/*
#include <opencv2\opencv.hpp>
#include <opencv2\contrib\contrib.hpp>
using namespace cv;

int main()
{
	//读取摄像头和视频
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		return -1;
	}
	Mat frame;
	Mat edges;
	bool stop = false;
	while (!stop)
	{
		capture >> frame;
		//cvtColor(frame, edges, CV_BGR2GRAY);
		//GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		//Canny(edges, edges, 0, 30, 3);
		imshow("当前视频", frame);
		if (waitKey(30) >= 0) {
			stop = true;
		}
	}

	return 0;
}
*/



/*
#include <iostream>
#include <stdio.h>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
using namespace std;
using namespace cv;


int main(int argc, char** argv)
{

	CascadeClassifier stFaceCascade;
	IplImage *pstImage = NULL;
	std::vector faceRects;
	if (!stFaceCascade.load("D:\opencv\sources\data\haarcascades\haarcascade_eye_tree_eyeglasses.xml"))
	{
		printf("Loading cascade error\n");
		return -1;
	}

	pstImage = cvLoadImage("", CV_LOAD_IMAGE_COLOR);

	stFaceCascade.detectMultiScale(pstImage,
		faceRects,            //检出结果
		1.1,                  //缩放步长
		2,                    //框融合时的最小检出个数
		0 | CV_HAAR_SCALE_IMAGE,//标志 |CV_HAAR_FIND_BIGGEST_OBJECT|CV_HAAR_DO_ROUGH_SEARCH|CV_HAAR_DO_CANNY_PRUNING
		Size(30, 30),         //最小人脸尺寸
		Size(300, 300));     //最大人脸尺寸
	printf("Face Num[%d]\n", faceRects.size());
	for (unsigned int j = 0; j < faceRects.size(); j++)
	{
		cvLine(pstImage,
			cvPoint(faceRects[j].x + faceRects[j].width / 2, faceRects[j].y + faceRects[j].height / 2),
			cvPoint(faceRects[j].x + faceRects[j].width / 2, faceRects[j].y + faceRects[j].height / 2),
			cvScalar(0, 255, 0),
			2, 8, 0);
	}
	cvShowImage("FDWin", pstImage);
	cvWaitKey(0);


	cvReleaseImage(&pstImage);
	return 0;
}*/









// opencv_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

/*
#include <opencv2/opencv.hpp>
#include <iostream>
int main()
{
	cv::Mat src = cv::imread("C:/Users/HZ_007/Desktop/JT201911292011232.JPG");
	cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
	cv::imshow("impage", src);
	cv::waitKey(0);
    std::cout << "Hello World!\n";
	system("pause");
}
*/
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
