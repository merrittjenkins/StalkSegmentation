#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include <sstream>
#include <string>
#include <boost/make_shared.hpp>
#include <pcl/point_representation.h>

#include <iostream>
#include <cstdio>
#include <ctime>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/passthrough.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/conditional_removal.h>


#include "standard_includes.h"
#include "scale_image.h"
#include "write_ply.h"
#include "feature_match.h"


using namespace cv;
using namespace std;

using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// This is a tutorial so we can afford having global variables 
//our visualizer
pcl::visualization::PCLVisualizer *p;
//its left and right viewports
int vp_1, vp_2;

//pcl::PointCloud<pcl::PointXYZRGB>::Ptr print_message_function( void *ptr);
void *print_message_function( void *ptr);




pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_1 (new pcl::PointCloud<pcl::PointXYZRGB>);

main()
{
     std::clock_t start_global;
     start_global = std::clock();

     pthread_t thread1, thread2, thread3, thread4;
     char *message1 = "5";
     char *message2 = "6";
     char *message3 = "7";
     char *message4 = "8";
     int iret1, iret2, iret3, iret4;

    // Create independent threads each of which will execute function

     //pcl::PointCloud<pcl::PointXYZRGB>::Ptr iret1 (new PointCloud);

     iret1 = pthread_create( &thread1, NULL, print_message_function, (void*) message1);
     iret2 = pthread_create( &thread2, NULL, print_message_function, (void*) message2);
     iret3 = pthread_create( &thread3, NULL, print_message_function, (void*) message3);
     iret4 = pthread_create( &thread4, NULL, print_message_function, (void*) message4);

     // Wait till threads are complete before main continues. Unless we 
     // wait we run the risk of executing an exit which will terminate   
     // the process and all threads before the threads have completed. 

     pthread_join( thread1, NULL);
     pthread_join( thread2, NULL);
     pthread_join( thread3, NULL);
     pthread_join( thread4, NULL); 

     //printf("Thread 1 returns: %d\n",iret1);   
     //printf("Thread 2 returns: %d\n",iret2);
     //printf("Thread 3 returns: %d\n",iret3);
     //printf("Thread 4 returns: %d\n",iret4);

     //pcl::PLYWriter writer;
     //writer.write<pcl::PointXYZRGB>("tester.ply", *cloud_1, false);

     double feat_time = (clock() - start_global) / (double) CLOCKS_PER_SEC;
     cout << "TOTAL TIME: " << feat_time << endl;
     
     exit(0);
}

void *print_message_function( void *ptr)
{
     char *message;
     message = (char *) ptr;
     printf("%s \n", message);

     int w = atoi(message);

     float u;
     float v;

     //Mat disp, disp8;
     //Mat xyz, inliers;
     //int scale_factor=4;
     //std::vector< DMatch > good_matches;
     std::clock_t start;

     vector<Point3f> list_points3d;
     vector<Point2f> list_points2d;
     vector<KeyPoint> keypoints_im2;

     const char* point_cloud_filename = 0;
     const char* point_cloud_filename_2 = 0;
     int scale_factor=4;

     point_cloud_filename = "cloud.ply";

     std::vector< DMatch > good_matches;

     Mat disp, disp8;

     Mat descriptors_1, descriptors_2;

     Mat xyz, inliers;

     bool flags = 1;
     Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);
     Mat rvec = Mat::zeros(3, 1, CV_64FC1);
     Mat tvec = Mat::zeros(3, 1, CV_64FC1);

     Mat R_matrix = Mat::zeros(3, 3, CV_64FC1);
     Mat t_matrix = Mat::zeros(3, 1, CV_64FC1);

     bool useExtrinsicGuess = false;
     int iterationsCount=10000;
     float reprojectionError=5.0;
     double confidence=0.99;


     Mat jacobian;
     double aspectRatio = 0;
     vector<Point2f> imagePoints; 

     Point pt;
     vector<KeyPoint> keypoints_projected;

     Mat other, t_total;

     PointCloud::Ptr ptCloud (new PointCloud);
     PointCloud::Ptr cloud_filtered (new PointCloud);
     PointCloud::Ptr cloud_filtered_old (new PointCloud);
     int max_z = 1.0e4;

     //----------------------------------CONFIGURE STEREO AND CAMERA SETTINGS----------------------------------

     //assign SGBM because it gives superior results
     enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3, STEREO_3WAY=4 };
     int alg = STEREO_SGBM;

     int color_mode = alg == STEREO_BM ? 0 : -1;

     int SADWindowSize = 11;
     bool no_display;
     float scale;

     int minDisparity = 80;
     int numberOfDisparities = 224;
     int uniquenessRatio = 10;
     int speckleWindowSize = 10;
     int speckleRange = 1;
     int disp12MaxDiff = 10;
     int P1 = pow(8*3*SADWindowSize,2);
     int P2 = pow(32*3*SADWindowSize,2);
     bool fullDP=false;
     int preFilterCap = 0;
     int mode;

     //camera intrinsics of my specific camera
     float cx = 1688/scale_factor;
     float cy = 1352/scale_factor;
     float f = 2421/scale_factor;
     float T = -.0914*scale_factor; 

     // M1 is the camera intrinsics of the left camera (known)
     Mat M1 = (Mat_<double>(3,3) << 2421.988247695817/scale_factor, 0.0, 1689.668741757609/scale_factor, 0.0, 2424.953969600827/scale_factor, 1372.029058638022/scale_factor, 0.0, 0.0, 1.0);

     int minHessian = 800; //the hessian affects the number of keypoints

     Mat Q = (Mat_<double>(4,4) << -1,0,0,cx,0,1,0,-cy,0,0,0,f,0,0,1/T,0);

     int cx1;
     int cx2;
     int cx3;
     char buffer [100];
     char img1_filename[100];
     char img2_filename[100];
     char img3_filename[100];


     // Create a PCLVisualizer object
     //p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Incremental Registration example");
     //p->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
     //p->createViewPort (0.5, 0, 1.0, 1.0, vp_2);

     PointCloud::Ptr source, target;
     Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform;

     std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_vec;

     PointCloud::Ptr result (new PointCloud);


    //------------------------------------LOAD IMAGES-----------------------------------------
    //The first two images are the stereo pair. The third image is the one that has moved (we don't know its position)

    int offset = 0;
    int imstart = w + offset;

    if (imstart<9) {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images/left00000%d.jpg", imstart);
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images/right00000%d.jpg", imstart);
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images/left00000%d.jpg", imstart+1);}    
    else if (imstart==9) {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images/left00000%d.jpg", imstart);
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images/right00000%d.jpg", imstart);
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images/left0000%d.jpg", imstart+1);}     
    else {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images/left0000%d.jpg", imstart);  
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images/right0000%d.jpg", imstart); 
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images/left0000%d.jpg", imstart+1);}


    double feat_time;
    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, load images: " << feat_time << endl;

    //-------------------------------------------SCALE IMAGES-------------------------------------------

    scaleImage scaledown; //(img1_filename, color_mode);
    scaleImage scaledown2; //(img3_filename, color_mode);
    scaleImage scaledown3; //(img2_filename, color_mode);

    Mat img1 = scaledown.downSize(img1_filename, color_mode);
    Mat img2 = scaledown2.downSize(img2_filename, color_mode);
    Mat img3 = scaledown3.downSize(img3_filename, color_mode);

    Size img_size = img1.size();

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, downsample: " << feat_time << endl;

    //------------------------------------FIND MATCHES-----------------------------------------

    //-- Step 1: Detect keypoints

    SurfFeatureDetector detector(minHessian);

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img1, keypoints_1 );
    detector.detect( img3, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    //SurfDescriptorExtractor extractor;

    //extractor.compute( img1, keypoints_1, descriptors_1 );
    //extractor.compute( img3, keypoints_2, descriptors_2 );
    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, extract features: "<< feat_time << endl;

    //-- Step 3: Choose only the "good" matches
    featureMatch detect;
    good_matches = detect.goodmatchDetect(keypoints_1, keypoints_2, img1, img3);

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, feature matching: "<< feat_time << endl;

    //waitKey(0);


    //-----------------------------------------------CONVERT TWO IMAGES TO DISPARITY IMAGE-----------------------------------------------

    StereoSGBM sgbm(minDisparity, numberOfDisparities, SADWindowSize, P1, P2, disp12MaxDiff, preFilterCap,\
        uniquenessRatio, speckleWindowSize, speckleRange, fullDP);  

    //int64 t = getTickCount();

    // convert the two images into a disparity image
    sgbm(img1, img2, disp);
    //t = getTickCount() - t;
    //printf("SGBM time elapsed: %fms\n", t*1000/getTickFrequency());

    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else 
        disp.convertTo(disp8, CV_8U);

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, SGBM: "<< feat_time << endl;

    //---------------------------------------------------BUILD 3D CLOUD------------------------------------------------------    
    
    // Project to 3D, points filled into xyz mat format
    reprojectImageTo3D(disp8, xyz, Q, true);

        //This loops through all of the matched feature points and finds the image coordinate in 3D. 
    //If the z-coordinate is less than 3D (ie if the block matching found a match),
    //then the 3d points are saved to a vector and the 2D feature points in the second image
    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        u = keypoints_1[good_matches[i].queryIdx].pt.x;
        v = keypoints_1[good_matches[i].queryIdx].pt.y;
        Vec3f point = xyz.at<Vec3f>(v, u);
        if (point[2]<10000)
        {
            list_points3d.push_back(Point3f(point[0],point[1],point[2]));
            list_points2d.push_back(Point2f(keypoints_2[good_matches[i].trainIdx].pt.x, keypoints_2[good_matches[i].trainIdx].pt.y));
            keypoints_im2.push_back(KeyPoint(keypoints_2[good_matches[i].trainIdx]));
        }
    }

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, build cloud and features: "<< feat_time << endl;

//----------------------------------------------------------SOLVE PNP------------------------------------------------------

    int64 t_pnp = getTickCount();

    solvePnPRansac( list_points3d, list_points2d, M1, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags);
    
    Rodrigues(rvec,R_matrix);
    t_matrix = tvec;

    //cout << "Rotation matrix: " << R_matrix << endl;
    //cout << "Translation matrix: " << t_matrix << endl;
    //cout << "\n"<<"Number of inliers: " << inliers.rows << endl;

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, solve PnP: "<< feat_time << endl;
  
//---------------------------------------------CONVERT POINT CLOUD TO PCL FORMAT-------------------------------------------


    for(int row = 0; row < xyz.rows; row++)
    {
        for(int col = 0; col < xyz.cols; col++)
        {
            Vec3f point = xyz.at<Vec3f>(row,col);
            Vec3b intensity = img1.at<Vec3b>(row,col);
            uchar blue = intensity.val[0];
            uchar green = intensity.val[1];
            uchar red = intensity.val[2];
            //if (row<5 && col<5){cout << point[0] << ", " << point[1] << ", " << point[2] << endl;}

            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            //i++;
            //cout << i << endl;
            //pcl::PointCloud<PointT> point_loc;
            PointT point_loc;
            point_loc.z = point[2];
            point_loc.x = point[0];
            point_loc.y = point[1];
            point_loc.r = red;
            point_loc.g = green;
            point_loc.b = blue;
            ptCloud->points.push_back(point_loc);
        }
    }

    //ptCloud->width = (int)xyz.cols; 
    //ptCloud->height = (int)xyz.rows; 

    // end clock, for kicks
    //duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    //std::cout<< "\n" <<"Total time: "<< duration <<'\n';

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, convert cloud to PCL: "<< feat_time << endl;

    //stringstream ss;
    //ss.

    copyPointCloud(*ptCloud, *cloud_1);

    //return(ptCloud);
}