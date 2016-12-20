/*
#include<stdio.h>
#include<math.h>*/
#include<opencv\cv.h>
#include<opencv\highgui.h>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core.hpp"
/*#include<vector>
#include<fstream>
#include <iostream>
#include <sstream>
#include <string>*/
#include<bits/stdc++.h>


using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void readCSV1(istream &input, vector< vector<string> > &output)
{
   string csvLine;
    // read every line from the stream
    while( getline(input, csvLine) )
    {
            istringstream csvStream(csvLine);
           vector<string> csvColumn;
            string csvElement;
            // read every element from the line that is seperated by commas
            // and put it into the vector or strings
            while( getline(csvStream, csvElement, ',') )
            {
                    csvColumn.push_back(csvElement);
            }
            output.push_back(csvColumn);
    }
}

int main()
{
    int count =1;
    int cnt,m=0;
	CascadeClassifier face_cascade, face_cascade1;
	if(!face_cascade.load("C:/opencv/opencv/sources/data/Haarcascades/haarcascade_frontalface_alt2.xml")) {
		printf("Error loading cascade file for face");
		return 1;
	}
    IplImage* img = cvLoadImage("C:/Users/radhika/Desktop/aa.jpg", 1);
    Mat cap_img(img);

    //resize the picture
    const int DETECTION_WIDTH = 1000;
    // Possibly shrink the image, to run much faster.
    Mat smallImg;

    float scale = cap_img.cols / (float) DETECTION_WIDTH;
    int scaledHeight = cvRound(cap_img.rows / scale);
    resize(cap_img, smallImg, Size(DETECTION_WIDTH, scaledHeight));

    cap_img=smallImg;
    //end of resizing

    //convert to gray image
	Mat gray_img;


	vector<Rect> faces;

        cvtColor(cap_img, gray_img, CV_BGR2GRAY);
        Mat gray_1=gray_img;
		cv::equalizeHist(gray_img,gray_img);
		Mat Hist1=gray_img;


		face_cascade.detectMultiScale(gray_img, faces, 1.1, 2.5, 0,  cvSize(0,0), cvSize(300,300));

		for(int i=0; i < faces.size();i++)
		{
			Point pt1(faces[i].x+faces[i].width, faces[i].y+faces[i].height);
			Point pt2(faces[i].x,faces[i].y);
			//Mat faceROI = gray_img(faces[i]);
            rectangle(cap_img, pt1, pt2, cvScalar(0,255,0), 2, 8, 0);

                Rect roi = faces[i];
               // Cut it out of the original image
               Mat face = gray_img(roi).clone();

               Mat fac;
               const int WIDTH = 50;
                float scal = face.cols / (float) WIDTH;
                int scaledHeit = cvRound(face.rows / scal);
                resize(face, fac, Size(WIDTH, scaledHeit));

               // Store the image

               string str="C:/Users/radhika/Desktop/cc/";

                char numstr[20];
                sprintf(numstr, "%d", count);
                count++;
                string result;
                result=str+numstr+".pgm";

               imwrite(result, fac);


		}

		 ofstream myfile;

    myfile.open("C:/Users/radhika/Desktop/detected_faces.csv");
     string str = "C:/Users/radhika/Desktop/cc/";

      char numstr[20];
      for(int i=1;i<count;i++)
      {
                sprintf(numstr, "%d", i);
                string result;
                result=str+numstr+".pgm";
    myfile<<result;
     myfile<<";";
    myfile<<i;
    myfile<<";";
    myfile<<"\n";
      }
    myfile.close();
    imshow("detected faces", cap_img);


    //face recognition code here
    int label_arr[10];
    int preval;

     // Get the path to your CSV.
    string fn_csv = "C:/Users/radhika/Desktop/final.csv";
     string fn_csv1 = "C:/Users/radhika/Desktop/detected_faces.csv";
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<Mat> images1;
    vector<int> labels;
    vector<int> labels1;
    // Read in the data. This can fail if no valid
    // input filename is given.

    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

     try {
        read_csv(fn_csv1, images1, labels1);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }


    // Quit if there are not enough images for this demo.
    if(images.size() <1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

 for(int i=0;i<images1.size();i++)
    {
    Mat testSample = images1[i];
    int testLabel = labels1[i];

    // The following lines create an Eigenfaces model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.

    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);
    // The following line predicts the label of a given
    // test image:

     int predictedLabel = -1;
        double confidence = 0.0;
       model->predict(testSample, predictedLabel, confidence);


    //int predictedLabel = model->predict(testSample);
    //cout<<confidence<<endl;
    if(confidence < 1000.0)
    {
    label_arr[m++]=predictedLabel;
    }

    }

    cout<<m<<" students are present in the classroom"<<endl;



    ofstream myfile3;
    string a;
    int flag=0;
    int p=1;
    int x=0;
    fstream file("C:/Users/radhika/Desktop/b.csv", ios::in);
    myfile3.open ("C:/Users/radhika/Desktop/ab.csv");
    if(!file.is_open())
    {
           cout << "File not found!\n";
            return 1;
    }
    // typedef to save typing for the following object
    typedef vector< vector<string> > csvVector;
    csvVector csvData;

    readCSV1(file, csvData);
    // print out read data to prove reading worked
    for(csvVector::iterator i = csvData.begin(); i != csvData.end(); ++i)
    {
            for(vector<string>::iterator j = i->begin(); j != i->end(); ++j)
            {
          a=*j;
            for(int i=0;i<m;i++)
            {
                x=0;
                stringstream convert(a);//object from the class stringstream
                convert>>x;
                if(x==label_arr[i])
                {
                    flag=1;
                    break;

                }
            }

        if(flag==1 && p!=0)
        {

            cout << a << " ";
            myfile3 <<a<<",";
            p=0;
        }
        else if(flag==0){
                cout <<a<< " ";
                myfile3 <<a<<",";

        }
        else if(p==0)
        {

            cout <<"present"<< " ";
          myfile3 <<"present"<<",";
          flag=0;
          p=1;

        }
}
 myfile3 <<"\n";
 cout << "\n";
 }
 myfile3.close();


    waitKey(0);
    return 0;
}



