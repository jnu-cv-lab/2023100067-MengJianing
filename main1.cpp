#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    //  任务1：读取 PNG 测试图片 
    Mat img = imread("text.png", IMREAD_COLOR); 
    if (img.empty()) {
        cout << " 错误：找不到 text.png" << endl;
        return -1;
    }

    // 任务2：输出图像基本信息
    cout << "\n 图像基本信息" << endl;
    cout << " 宽度: " << img.cols << " px" << endl;
    cout << " 高度: " << img.rows << " px" << endl;
    cout << " 通道数: " << img.channels() << endl;
    
    string img_type;
    switch (img.type()) {
        case CV_8UC1: img_type = "CV_8UC1 (8位单通道)"; break;
        case CV_8UC3: img_type = "CV_8UC3 (8位3通道/BGR)"; break;
        default: img_type = "其他类型"; break;
    }
    cout << "数据类型: " << img_type << endl;
    cout << "\n" << endl;

    // 任务3：显示原图 
    namedWindow(" 原图", WINDOW_AUTOSIZE);
    imshow("原图", img);

    //  任务4：转灰度图并显示 
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray_img);

    //  任务5：保存灰度图
    imwrite("gray_text.png", gray_img);
    cout << " 灰度图已保存: gray_text.png" << endl;

    //  任务6：NumPy 风格操作（用 OpenCV Mat 实现） 
    cout << "\n像素操作" << endl;
    // 输出左上角 (0,0) 像素值
    if (img.channels() == 3) {
        Vec3b pixel = img.at<Vec3b>(0, 0);
        cout << " 左上角像素 (BGR): " << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << endl;
    } else {
        uchar pixel = img.at<uchar>(0, 0);
        cout << " 左上角像素: " << (int)pixel << endl;
    }

    // 裁剪左上角 100x100 区域并保存
    Rect roi(0, 0, 100, 100);
    Mat crop_img = img(roi);
    imwrite("crop_text.png", crop_img);
    cout << "裁剪区域已保存: crop_text.png" << endl;
    cout << "\n" << endl;

    // 等待按键关闭窗口
    waitKey(0);
    destroyAllWindows();

    return 0;
}