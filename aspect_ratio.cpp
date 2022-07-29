#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_transforms.h>

#define SHOW 1

using namespace dlib;
using namespace std;

int main() {
  std::vector<uchar> date_encode;
  float d, b, c, e, f;
  time_t start = time(NULL);
  cout << "开始时间start" << start << endl;
  char strTime[100], time_tp[10];
  string image_filename;
  string save_image_path = "image/";  //保存截图路径

  float eyesleep = 0.23;
  int COUNTER = 0;
  int EYE_ASPECT_RATIO_THRESHOLD = 150;
  b = sqrt(pow(339 - 340, 2) + pow(338 - 344, 2));
  printf("%f", b);
  clock_t mid;
  try {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      cerr << "无法连接到相机" << endl;
      return 1;
    }
    image_window win;
    // 加载面检测和姿态估计模型。人脸分类器
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    //获取人脸检测器
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    //抓取并处理帧，直到用户关闭主窗口。
    while (!win.is_closed()) {
      int tp_count = 0;
      // Grab a frame
      cv::Mat temp;
      if (!cap.read(temp)) {
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
      cout << "---=-=-=-=" << temp.rows << "-=-=" << temp.cols
           << endl;  //图像的宽高  480  640
      // 检测面
      std::vector<rectangle> faces = detector(cimg);
      //找出每一张脸的姿势
      // cout << sizeof(faces) << endl;
      cout << faces.size() << endl;
      std::vector<full_object_detection> shapes;
      for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));
      cout << shapes.empty() << endl;
      if (!shapes.empty()) {
        double scale = 2.;
        int LEFT_EYE = 42;
        double center_x = 0.5 * (shapes[0].part(LEFT_EYE + 3).x() +
                                 shapes[0].part(LEFT_EYE).x());
        double center_y = 0.5 * (shapes[0].part(LEFT_EYE + 5).y() +
                                 shapes[0].part(LEFT_EYE + 1).y());
        double x_scale =
            scale * 0.5 *
            (shapes[0].part(LEFT_EYE + 3).x() - shapes[0].part(LEFT_EYE).x());
        double y_scale = scale * 0.5 *
                         (shapes[0].part(LEFT_EYE + 5).y() -
                          shapes[0].part(LEFT_EYE + 1).y());
        cout << center_x << "__" << center_y << "__" << x_scale << "__"
             << y_scale << "__" << endl;
        std::array<dpoint, 4> LEFT_ROI;

        LEFT_ROI[0](0) = center_x - x_scale;
        LEFT_ROI[0](1) = center_y + y_scale;
        // the topleft point
        LEFT_ROI[1](0) = LEFT_ROI[0](0);
        LEFT_ROI[1](1) = center_y - y_scale;
        // the topright point
        LEFT_ROI[2](0) = center_x + x_scale;
        LEFT_ROI[2](1) = LEFT_ROI[1](1);
        // the bottomright point
        LEFT_ROI[3](0) = LEFT_ROI[2](0);
        LEFT_ROI[3](1) = LEFT_ROI[0](1);

#if SHOW
        cv::Mat cv_left_roi = toMat(left_roi);
        cv::namedWindow("left_roi");
        cv::imshow("left_roi", cv_left_roi);
        cv::waitKey(20);

#endif
        cout << "LEFT_ROI[0]" << LEFT_ROI[0] << endl;
        cout << "LEFT_ROI[1]" << LEFT_ROI[1] << "LEFT_ROI[2]" << LEFT_ROI[2]
             << endl;
      }

      if (!shapes.empty())
        for (int i = 42; i < 48; i++) {  //左眼36-41   右眼42-47
          stringstream ss;
          ss << 1;
          string str = ss.str();

          int image_size = temp.cols * temp.rows;
          unsigned char* imageData = new unsigned char[image_size];

          // circle(temp, cvPoint(shapes[0].part(i).x(), shapes[0].part(i).y()),
          // 3, cv::Scalar(0, 0, 255), -1); cv::putText(temp,str,
          // cvPoint(shapes[0].part(i).x(),
          // shapes[0].part(i).y()),cv::FONT_HERSHEY_SCRIPT_SIMPLEX,
          // 1,cv::Scalar(0,0,255), 1, 1);
          cv::putText(temp, str,
                      cvPoint(shapes[0].part(42).x(), shapes[0].part(42).y()),
                      cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255),
                      1, 1);
          cout << i << "-----" << shapes[0].part(i).x() << endl;
          cout << shapes[0].part(i) << endl;
          c = sqrt(pow(shapes[0].part(43).x() - shapes[0].part(47).x(), 2) +
                   pow(shapes[0].part(43).y() - shapes[0].part(47).y(), 2));
          d = sqrt(pow(shapes[0].part(44).x() - shapes[0].part(46).x(), 2) +
                   pow(shapes[0].part(44).y() - shapes[0].part(46).y(), 2));
          e = sqrt(pow(shapes[0].part(42).x() - shapes[0].part(45).x(), 2) +
                   pow(shapes[0].part(42).y() - shapes[0].part(45).y(), 2));
          printf("c:%f--d:%f--e:%f", c, d, e);
          f = (c + d) / (2 * e);
          printf("--------%f", f);
          if (f < eyesleep) {
            COUNTER++;
            if (COUNTER >= EYE_ASPECT_RATIO_THRESHOLD) {
              tp_count++;
              cv::putText(
                  temp, "You are Drowsy",
                  cvPoint(shapes[0].part(42).x(), shapes[0].part(42).y()),
                  cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1,
                  1);  //左眼闭眼提示
              start = time(NULL);

              strftime(strTime, sizeof(strTime), "%Y%m%d%H%M%S",
                       localtime(&start));
              strftime(time_tp, sizeof(time_tp), "%S", localtime(&start));
              image_filename = save_image_path + "JT" + strTime +
                               to_string(tp_count) + ".JPG";
              // cv::putText(cv_img, "You are Drowsy", cv::Point(20, 20),
              // cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1,
              // 1);
              cv::imwrite(image_filename, temp);

              cv::imencode(".jpg", temp, date_encode);
              std::string str_encode(date_encode.begin(), date_encode.end());
            }
          } else {
            start = time(NULL);
            COUNTER = 0;
            tp_count = 0;
          }
        }

        //全部显示在屏幕上dlib显示
#if SHOW
      win.clear_overlay();
      win.set_image(cimg);
      win.add_overlay(render_face_detections(shapes));
#endif
    }
  } catch (serialization_error& e) {
    cout << "You need  model file to run this example." << endl;
  } catch (exception& e) {
    cout << e.what() << endl;
  }
}