/* snprintf example */
#include <stdio.h>
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


/* ---[ */
int main (int argc, char** argv)
{

  float u;
  float v;

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
  int numberOfDisparities = 320; //224;
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
  float cx = 2120/scale_factor;
  float cy = 1412/scale_factor;
  //float f = 2421/scale_factor;
  float f = 2892/scale_factor;  //I just pulled this from M1 (as of Feb 7th)...better check
  float T = -.0914*scale_factor; //I can't remember what this does...

  // M1 is the camera intrinsics of the left camera (known)
  //Mat M1 = (Mat_<double>(3,3) << 2421.988247695817/scale_factor, 0.0, 1689.668741757609/scale_factor, 0.0, 2424.953969600827/scale_factor, 1372.029058638022/scale_factor, 0.0, 0.0, 1.0);
  Mat M1 = (Mat_<double>(3,3) << 2892.056424058859/scale_factor, 0.0, 1406.936028219495/scale_factor, 0.0, 2887.430090443049/scale_factor, 2206.893298336217/scale_factor, 0.0, 0.0, 1.0);

  int minHessian = 200; //the hessian affects the number of keypoints

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

  int imstart1= 14;
  int imstart2= 16;  

  cx1 = snprintf ( img1_filename, 100, "../bagfiles/row_10.5_S2Na/left0000%d.jpg", imstart1);
  cx2 = snprintf ( img2_filename, 100, "../bagfiles/row_10.5_S2Na/right0000%d.jpg", imstart2);

  //cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images/left000006.jpg");
  //cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images/right000006.jpg");

  scaleImage scaledown; //(img1_filename, color_mode);
  scaleImage scaledown2; //(img3_filename, color_mode);

  Mat img1 = scaledown.downSize(img1_filename, color_mode);
  Mat img2 = scaledown2.downSize(img2_filename, color_mode);

  //Mat img1 = imread("../bagfiles/row_10.5_S2Na/left000009.jpg", CV_LOAD_IMAGE_COLOR);
  //Mat img2 = imread("../bagfiles/row_10.5_S2Na/right000011.jpg", CV_LOAD_IMAGE_COLOR);

  Size img_size = img1.size();

  imwrite( "../bagfiles/row_10.5_S2Na/leftmatch.jpg", img1);
  imwrite( "../bagfiles/row_10.5_S2Na/leftmatch.jpg", img2);  
  
  //-----------------------------------------------CONVERT TWO IMAGES TO DISPARITY IMAGE-----------------------------------------------

  StereoSGBM sgbm(minDisparity, numberOfDisparities, SADWindowSize, P1, P2, disp12MaxDiff, preFilterCap,\
      uniquenessRatio, speckleWindowSize, speckleRange, fullDP);  

  int64 t = getTickCount();

  // convert the two images into a disparity image
  sgbm(img1, img2, disp);
  t = getTickCount() - t;
  printf("SGBM time elapsed: %fms\n", t*1000/getTickFrequency());

  if( alg != STEREO_VAR )
      disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
  else 
      disp.convertTo(disp8, CV_8U);

  namedWindow("disparity", 0);
  imshow("disparity", disp8);
  printf("press any key to continue...");
  fflush(stdout);
  waitKey();
  printf("\n");


// Project to 3D, points filled into xyz mat format
  reprojectImageTo3D(disp8, xyz, Q, true);

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

    pcl::PLYWriter writer;
    writer.write<pcl::PointXYZRGB> (point_cloud_filename, *ptCloud, false);


  return(0);
}