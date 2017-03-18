/* snprintf example */
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

//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};

struct PCDComparator
{
  bool operator () (const PCD& p1, const PCD& p2)
  {
    return (p1.f_name < p2.f_name);
  }
};


// Define a new point representation for < x, y, z, curvature >
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT>
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
public:
  MyPointRepresentation ()
  {
    // Define the number of dimensions
    nr_dimensions_ = 4;
  }

  // Override the copyToFloatArray method to define our feature vector
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    // < x, y, z, curvature >
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the first viewport of the visualizer
 *
 */
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target");
  p->removePointCloud ("vp1_source");

  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0);
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1);
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);

  PCL_INFO ("Press q to begin the registration.\n");
  p-> spin();
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Display source and target on the second viewport of the visualizer
 *
 */
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source");
  p->removePointCloud ("target");


  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature");
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");

  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature");
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");


  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2);
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);

  p->spinOnce();
}

////////////////////////////////////////////////////////////////////////////////
/** \brief Load a set of PCD files that we want to register together
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd");
  // Suppose the first argument is the actual test model
  for (int i = 1; i < argc; i++)
  {
    std::string fname = std::string (argv[i]);
    // Needs to be at least 5: .plot
    if (fname.size () <= extension.size ())
      continue;

  //UNCLEAR TO MERRITT WHAT THIS IS/DOES
    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower);

    //check that the argument is a pcd file
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // Load the cloud and saves it into the global list of models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);

      models.push_back (m);
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/** \brief Align a pair of PointCloud datasets and return the result
  * \param cloud_src the source PointCloud
  * \param cloud_tgt the target PointCloud
  * \param output the resultant aligned source PointCloud
  * \param final_transform the resultant transform between source and target
  */
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{

  //
  // Downsample for consistency and speed
  // \note enable this for large datasets

  std::clock_t start;
  start = std::clock();
  double duration;


  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid;
  if (downsample)
  {
    grid.setLeafSize (0.03, 0.03, 0.03);
    grid.setInputCloud (cloud_src);
    grid.filter (*src);

    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else
  {
    src = cloud_src;
    tgt = cloud_tgt;
  }

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout<<"------TIME, voxellize cloud: " << duration << endl;

  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);
  
  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout<<"------TIME, estimate normals: " << duration << endl;

  //
  // Instantiate our custom point representation (defined above) ...
  MyPointRepresentation point_representation;
  // ... and weight the 'curvature' dimension so that it is balanced against x, y, and z
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha);

  //
  // Align
  pcl::IterativeClosestPoint<PointNormalT, PointNormalT> reg;
  reg.setTransformationEpsilon (1e-6);
  // Set the maximum distance between two correspondences (src<->tgt) to 10cm
  // Note: adjust this based on the size of your datasets
  reg.setMaxCorrespondenceDistance (0.05);  //originally 0.15
  // Set the point representation
  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation));

  reg.setInputSource (points_with_normals_src);
  reg.setInputTarget (points_with_normals_tgt);



  //
  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src;
  reg.setMaximumIterations (2);

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout<<"------TIME, set ICP variables: " << duration << endl;

  //for (int i = 0; i < 100; ++i)

  int i=0;
  while (reg.getMaxCorrespondenceDistance() > 0.01)
  {
    i = i+1;
    //PCL_INFO ("Iteration Nr. %d.\n", i);

    // save cloud for visualization purpose
    points_with_normals_src = reg_result;

    // Estimate
    reg.setInputSource (points_with_normals_src);
    reg.align (*reg_result);

    //accumulate transformation between each Iteration
    Ti = reg.getFinalTransformation () * Ti;

    //if the difference between this transformation and the previous one
    //is smaller than the threshold, refine the process by reducing
    //the maximal correspondence distance
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
      reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.01);

    prev = reg.getLastIncrementalTransformation ();

    // visualize current state
    //showCloudsRight(points_with_normals_tgt, points_with_normals_src);   
  }

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout<<"------TIME, actual ICP: " << duration << endl;

  cout << "Iterations: " << i << endl;

  // Get the transformation from target to source
  targetToSource = Ti.inverse();

  //
  // Transform target back in source frame
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);

  //cout << "Removing source and target clouds" << endl;

  //p->removePointCloud ("source");
  //p->removePointCloud ("target");

  //cout << "Removed source and target clouds" << endl;

  PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0);
  PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
  //p->addPointCloud (output, cloud_tgt_h, "target", vp_2);
  //p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);
  //PCL_INFO ("Press q to continue the registration.\n");
  //p->spin ();

  //p->removePointCloud ("source"); 
  //p->removePointCloud ("target");

  //add the source to the transformed target
  *output += *cloud_src;
  
  final_transform = targetToSource;

  duration = (clock() - start) / (double) CLOCKS_PER_SEC;
  cout<<"------TIME, end of ICP function: " << duration << endl;
 }

pcl::PointCloud<pcl::PointNormal>::Ptr filterFunction(pcl::PointCloud<pcl::PointNormal>::Ptr cloud_up, pcl::PointXYZ maxPt, pcl::PointXYZ minPt, float extension)
{
//-------------------Apply filtering based on normals and x-density-----------------------

  float z_distr = maxPt.z-minPt.z+2*extension;
  float y_distr = maxPt.y-minPt.y+2*extension;

  //cout << "number of points: " << cloud_up->size() << endl;
  ////cout << "z_distr is: " << z_distr << endl;
  ////cout << "y_distr is: " << y_distr << endl;

  int no_of_rows = int(y_distr*100)+1; 
  int no_of_cols = int(z_distr*100)+1;
  int initial_value = 0;

  ////cout << "boundary is: " << no_of_rows << ", " << no_of_cols << endl;

  std::vector<std::vector<int> > matrix;
  std::vector<std::vector<float> > matrix_normals;
  std::map <std::string, std::vector<int> > point_indices; //this is a map that will be filled with a string of the matrix location and the associated point indices

  matrix.resize(no_of_rows, std::vector<int>(no_of_cols, initial_value));
  matrix_normals.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));

  // Loop through every point in the cloud  
  for(int i=0; i<cloud_up->size(); i++)
  {
    int scatter_y = -int(((*cloud_up)[i].y*-100) + (minPt.y - extension)*100); //this is the y-location of the point
    int scatter_z = -int(((*cloud_up)[i].z*-100) + (minPt.z - extension)*100); //this is the z-location of the point

    matrix[scatter_y][scatter_z] = matrix[scatter_y][scatter_z]+1; //add a count to that cell location
    matrix_normals[scatter_y][scatter_z] = matrix_normals[scatter_y][scatter_z]+std::abs((*cloud_up)[i].normal_x); //add z-normal to that cell location

    //HERE I KEEP TRACK OF WHICH POINT INDICES ARE ASSOCIATED WITH WHICH LOCATION
    std::stringstream ss;
    ss << scatter_y << scatter_z;
    point_indices[ss.str()].push_back(i);
  }


  //float (*)(const char*) test = nanf;
  float nanInsert = std::numeric_limits<float>::quiet_NaN();

  // THIS TAKES THE AVERAGE NORMAL VECTOR FOR EACH VOXEL AND THEN DECIDES TO KEEP OR TOSS THE ELEMENTS OF THE VOXEL
  for(int i=0; i<no_of_rows; i++)
  {
    for(int j=0; j<no_of_cols; j++) 
    {
      if ((matrix_normals[i][j]) > 0)
      {
        matrix_normals[i][j] = (matrix_normals[i][j])/(matrix[i][j]);
      }

      if ((matrix[i][j]<30) || (matrix_normals[i][j]>0.5))
      {
        std::stringstream ss;
        ss << i << j;   

        // Iterate through all elements associated with the key and delete them
        for(int k=0; k<point_indices[ss.str()].size(); k++){  //assigning these as 1 is a total hack and needs to be fixed
          (*cloud_up)[point_indices.find(ss.str())->second[k]].normal_x = nanInsert;
        }
      }
    }
  }

  std::vector<int> indices2;
    pcl::removeNaNNormalsFromPointCloud(*cloud_up,*cloud_up, indices2);


    //THIS REGION-GROWS THE SLICE AND ONLY KEEPS THE LARGEST REGION
    pcl::VoxelGrid<pcl::PointNormal> vg;
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_vox (new pcl::PointCloud<pcl::PointNormal>);
  vg.setInputCloud (cloud_up);
  vg.setLeafSize (0.002f, 0.002f, 0.002f);
  vg.filter (*cloud_vox);

  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_vox);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointNormal> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree2);
  ec.setInputCloud (cloud_vox);
  ec.extract (cluster_indices);

  int j = 0;
  ////cout << "Size of cluster indices: " << cluster_indices.size() << endl;

  pcl::PointCloud<pcl::PointNormal>::Ptr cluster_saver (new pcl::PointCloud<pcl::PointNormal>);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointNormal>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_vox->points[*pit]); //*
    //cout << "Size of cluster: " << cloud_cluster->size() << endl;
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    ////cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << endl;
    
    if(it==cluster_indices.begin ())
      cluster_saver = cloud_cluster;
    else if(cloud_cluster->points.size() > cluster_saver->points.size())
      cluster_saver = cloud_cluster;
  }


    return(cloud_up);
}


pcl::PointCloud<pcl::PointNormal>::Ptr filterFunction2(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
{
//-------------------Apply filtering based on normals and x-density-----------------------

  pcl::PointNormal minPt, maxPt;
  pcl::getMinMax3D (*cloud, minPt, maxPt);

  float z_distr = maxPt.z-minPt.z;
  float y_distr = maxPt.y-minPt.y;

  //cout << "number of points: " << cloud_up->size() << endl;
  ////cout << "z_distr is: " << z_distr << endl;
  ////cout << "y_distr is: " << y_distr << endl;

  int no_of_rows = int(y_distr*100)+1; 
  int no_of_cols = int(z_distr*100)+1;
  int initial_value = 0;

  ////cout << "boundary is: " << no_of_rows << ", " << no_of_cols << endl;

  std::vector<std::vector<int> > matrix;
  std::vector<std::vector<float> > matrix_normals;
  std::map <std::string, std::vector<int> > point_indices; //this is a map that will be filled with a string of the matrix location and the associated point indices

  matrix.resize(no_of_rows, std::vector<int>(no_of_cols, initial_value));
  matrix_normals.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));

  // Loop through every point in the cloud  
  for(int i=0; i<cloud->size(); i++)
  {
    int scatter_y = -int(((*cloud)[i].y*-100) + (minPt.y)*100); //this is the y-location of the point
    int scatter_z = -int(((*cloud)[i].z*-100) + (minPt.z)*100); //this is the z-location of the point

    matrix[scatter_y][scatter_z] = matrix[scatter_y][scatter_z]+1; //add a count to that cell location
    matrix_normals[scatter_y][scatter_z] = matrix_normals[scatter_y][scatter_z]+std::abs((*cloud)[i].normal_x); //add z-normal to that cell location

    //HERE I KEEP TRACK OF WHICH POINT INDICES ARE ASSOCIATED WITH WHICH LOCATION
    std::stringstream ss;
    ss << scatter_y << scatter_z;
    point_indices[ss.str()].push_back(i);
  }


  float nanInsert = std::numeric_limits<float>::quiet_NaN();

  // THIS TAKES THE AVERAGE NORMAL VECTOR FOR EACH VOXEL AND THEN DECIDES TO KEEP OR TOSS THE ELEMENTS OF THE VOXEL
  for(int i=0; i<no_of_rows; i++)
  {
    for(int j=0; j<no_of_cols; j++) 
    {
      if ((matrix_normals[i][j]) > 0)
      {
        matrix_normals[i][j] = (matrix_normals[i][j])/(matrix[i][j]);
      }

      if ((matrix[i][j]<30) || (matrix_normals[i][j]>0.5))
      {
        std::stringstream ss;
        ss << i << j;   

        // Iterate through all elements associated with the key and delete them
        for(int k=0; k<point_indices[ss.str()].size(); k++){  //assigning these as 1 is a total hack and needs to be fixed
          (*cloud)[point_indices.find(ss.str())->second[k]].normal_x = nanInsert;
        }
      }
    }
  }

  std::vector<int> indices2;
    pcl::removeNaNNormalsFromPointCloud(*cloud,*cloud, indices2);

  return(cloud);
}


std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> regionGrow_ground(pcl::PointCloud<pcl::PointNormal>::Ptr cloud) {
  // Initialize a vector of cloud pointers
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> multiCloud;

  //Copy the incoming cloud to be PointXYZ instead of PointNormal
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_slice_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*cloud, *cloud_slice_xyz);

  // Voxellize the cloud (region growing doesn't seem to work otherwise)
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud_slice_xyz);
  vg.setLeafSize (0.002f, 0.002f, 0.002f);
  vg.filter (*cloud_filtered);

  ////cout << "Size of filtered cloud: " << cloud_filtered->size() << endl;

  // Apply region growing
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ>);
  tree2->setInputCloud (cloud_filtered);
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree2);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  // Convert indices to clouds, and stuff them into the multiCloud vector
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    multiCloud.push_back(cloud_cluster);
  }

  return(multiCloud);
}







/* ---[ */
int main (int argc, char** argv)
{

  std::clock_t start_global;
  start_global = std::clock();

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
  //Mat rvec = (Mat_<double>(3,3) << 1,0,0,0,1,0,0,0,1);
  Mat t_matrix = Mat::zeros(3, 1, CV_64FC1);
  //Mat tvec = (Mat_<double>(3,1) << 0.04, -0.2, 0);

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

  for (int w=0; w<5; w++)
  {
    PointCloud::Ptr result (new PointCloud);
    //start a timer just for kicks
    clock_t start;
    double duration;
    double feat_time;
    start = clock();
    
    //cout << "" << endl;
    //cout << "" << endl;
    //cout << "Iteration: " << w << endl;

    //------------------------------------LOAD IMAGES-----------------------------------------
    //The first two images are the stereo pair. The third image is the one that has moved (we don't know its position)

    int offset = 20;
    int imstart = w + offset;

    if (imstart<9) {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images2/left_00000%d.jpg", imstart);
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images2/right_00000%d.jpg", imstart);
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images2/left_00000%d.jpg", imstart+1);}    
    else if (imstart==9) {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images2/left_00000%d.jpg", imstart);
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images2/right_00000%d.jpg", imstart);
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images2/left_0000%d.jpg", imstart+1);}     
    else {
      cx1 = snprintf ( img1_filename, 100, "../Sorghum_Stitching/images2/left_0000%d.jpg", imstart);  
      cx2 = snprintf ( img2_filename, 100, "../Sorghum_Stitching/images2/right_0000%d.jpg", imstart); 
      cx3 = snprintf ( img3_filename, 100, "../Sorghum_Stitching/images2/left_0000%d.jpg", imstart+1);}


    //std::string img1_filename = "../Sorghum_Stitching/images/left000007.jpg";
    //std::string img2_filename = "../Sorghum_Stitching/images/right000007.jpg";
    //std::string img3_filename = "../Sorghum_Stitching/images/left000008.jpg";
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
    SurfDescriptorExtractor extractor;

    extractor.compute( img1, keypoints_1, descriptors_1 );
    extractor.compute( img3, keypoints_2, descriptors_2 );

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

    //-------------------------------------------------FIND FEATURE POINTS IN 3D----------------------------------------------

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

    int flag = 0;
    int RANSACcounter= 0;
    while (flag == 0) {
      solvePnPRansac( list_points3d, list_points2d, M1, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, confidence, inliers, flags);
      
      Rodrigues(rvec,R_matrix);
      t_matrix = tvec;

      //cout << "Rotation matrix: " << R_matrix << endl;
      cout << "Translation matrix: " << t_matrix << endl;

      if ((std::abs(t_matrix.at<double>(2,0))>0.05 || std::abs(t_matrix.at<double>(0,0))>0.15 || std::abs(t_matrix.at<double>(1,0))>0.40) && (RANSACcounter<4))
      {
        cout << "\033[1;31mMATCH FAIL COUNTER\033[0m\n" << endl;
        RANSACcounter++;
      }
      else if (RANSACcounter==4){
        cout << "\033[1;31mIMAGE MATCHING FAILED\033[0m\n" << endl;   
        flag =1;     
      }
      else {flag = 1;}
    }

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
  
    //pcl::PLYWriter writer;
    //writer.write<pcl::PointXYZRGB> ("tester.ply", *ptCloud, false); 










//---------------------------------------------------OFFSET CLOUD AND FILTER-------------------------------------------
    if (w == 0)
      other = Mat::zeros(3, 1, CV_64FC1);

    Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

    // Define a translation of 2.5 meters on the x axis.
    transform_2.translation() << -other.at<double>(0,0), other.at<double>(0,1), other.at<double>(0,2);
    float theta=0.0;
    // The same rotation matrix as before; theta radians arround Z axis
    transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*ptCloud, *transformed_cloud, transform_2);


    //---------Create a conditional removal filter to remove black pixels---------
    int rMax = 255;
    int rMin = 15;
    int gMax = 255;
    int gMin = 15;
    int bMax = 255;
    int bMin = 15;
    pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, rMax)));
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::GT, rMin)));
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, gMax)));
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::GT, gMin)));
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::LT, bMax)));
    color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, bMin)));

    // build the filter
    pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem (color_cond);
    condrem.setInputCloud (transformed_cloud);
    condrem.setKeepOrganized(true);
    
    // apply filter
    condrem.filter (*cloud_filtered);

    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_filtered);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-1.6, -1.0);
    pass.filter (*cloud_filtered);

    //-----Create the statistical outlier filtering object-----
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    
    // set thresholds,tree size, and apply filter
    sor.setInputCloud (cloud_filtered);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
    //
    
    // remove the nans so we can perform more filtering later
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud_filtered,*cloud_filtered, indices);

    feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout<<"TIME, offset cloud and filter: "<< feat_time << endl;


//---------------------------------------------------ICP AND WRITE TO FILE-------------------------------------------

    //int count = 0;
    //if (w ==0){
    //  cloud_vec.push_back(cloud_filtered);
    //}
    if (w>0)
    {
      source = cloud_filtered_old;
      //cout << cloud_filtered_old << endl;
      target = cloud_filtered;

      // Add visualization data
      //showCloudsLeft(source, target);

      //PointCloud::Ptr temp (new PointCloud);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp (new pcl::PointCloud<pcl::PointXYZRGB>);
      PCL_INFO ("Aligning...\n"); //%s (%d) with %s (%d).\n", source.f_name.c_str (), source->points.size (), target.f_name.c_str (), target->points.size ());
      pairAlign (source, target, temp, pairTransform, true);
 
      feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;     
      cout<<"TIME, pairAlign: "<< feat_time << endl;
      //transform current pair into the global transform
      pcl::transformPointCloud (*temp, *result, GlobalTransform);

      feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
      cout<<"TIME, transformPointCloud: "<< feat_time << endl;

      //update the global transform
      GlobalTransform = GlobalTransform * pairTransform;

      cloud_vec.push_back(result);
    }

    copyPointCloud(*cloud_filtered, *cloud_filtered_old); 

    if (w==0)
      t_total = Mat::zeros(3, 1, CV_64FC1);
    else
      t_total = other;
      other = t_total+t_matrix;
    //cout << "other: " << other << endl;

    ptCloud->clear();
    //feat_time = (clock() - start) / (double) CLOCKS_PER_SEC;
    //cout<<"TIME, ICP: "<< feat_time << endl;
  }


  //cout << "Cloud_vec size: " << cloud_vec.size() << endl;

  pcl::PLYWriter writer;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_a_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b (new pcl::PointCloud<pcl::PointXYZRGB>);

  cout << "SIZE OF CLOUDVEC: " << cloud_vec.size() << endl;

  for (int i = 0; i < cloud_vec.size(); i++)
  {
    //std::stringstream ss;
    //ss << "wtf_" << i << ".ply";
    //writer.write<pcl::PointXYZRGB> (ss.str (), *cloud_vec[i], false);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud_vec[i]);
    sor.setLeafSize (0.006f, 0.006f, 0.006f);
    sor.filter (*cloud_a_filtered);

    std::cerr << "PointCloud after filtering: " << cloud_a_filtered->width * cloud_a_filtered->height 
     << " data points (" << pcl::getFieldsList (*cloud_a_filtered) << ")." << endl;

    stringstream ss5;
    pcl::PLYWriter writer;
    ss5 << i << ".ply";
    writer.write<pcl::PointXYZRGB> (ss5.str(), *cloud_a_filtered, false); 

    *cloud_b+=*cloud_a_filtered;
    cloud_a_filtered ->clear();
  }  

  //std::cerr << *cloud_b << std::endl;

  //writer.write<pcl::PointXYZRGB> ("Concatenate_BIG2.ply", *cloud_b, false); 








  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_b_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);

  //-------------------------Create a conditional removal filter to remove green pixels------------------------
  int rMax = 80;//108; //kept low to keep out orange from traffic cone
  int rMin = 50;//45;//22; //numbers lower than 50 will yield green leaves
  int gMax = 52;//80;
  int gMin = 16;//16;
  int bMax = 28; //45;//28//52; //this blue is actually the critical value
  int bMin = 0;//11;
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, rMax)));
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::GT, rMin)));
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, gMax)));
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::GT, gMin)));
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::LT, bMax)));
  color_cond->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, bMin)));

  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem (color_cond);
  condrem.setInputCloud (cloud_b);
  condrem.setKeepOrganized(true);

  // apply filter
  condrem.filter (*cloud_b_filtered);

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_b_filtered,*cloud_b_filtered, indices);


  boost::shared_ptr<vector<int> > indicesptr (new vector<int> (indices)); 
  // Extract fInliers from the input cloud
  pcl::ExtractIndices<pcl::PointXYZRGB> extract2 ;
  extract2.setInputCloud (cloud_b);
  extract2.setIndices (indicesptr);
  extract2.setNegative (true); // Removes part_of_cloud from full cloud  and keep the rest
  extract2.filter (*cloud_b); 


  //std::cerr << "Cloud after filtering: " << std::endl;
  //std::cerr << *cloud_a_filtered << std::endl;

  writer.write<pcl::PointXYZRGB> ("filtered_brown.ply", *cloud_b_filtered, false);

 //-----------------------------------------Initialize for RANSAC------------------------------------------
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);
  pcl::ExtractIndices<PointT> extract;


  //-----------------------------------------Perform RANSAC on Ground Plane------------------------------------------
  // Estimate point normals
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_b_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);

  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.05);
  seg.setInputCloud (cloud_b_filtered);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  std::cerr << "Plane coefficient a: " << coefficients_plane->values[0] << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud (cloud_b_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_plane);
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
  //writer.write ("ground_plane.ply", *cloud_plane, false);


  //-----------------------------------------Rotate Point Cloud Based on Ground Plane------------------------------------------
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  // Define a translation on the x axis.
  float theta=atan(-(coefficients_plane->values[1])/(coefficients_plane->values[0]));
  // The same rotation matrix as before; theta radians arround Z axis
  transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

  // Print the transformation
  printf ("\nMethod #2: using an Affine3f\n");
  std::cout << transform.matrix() << std::endl;

  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::transformPointCloud(*cloud_plane, *cloud_plane, transform);
  //writer.write ("ground_plane_tform.ply", *cloud_plane, false);


  //----------------------------------Find the ground plane again and translate in X-----------------------------------
  //-------------------------------------------------(BIG HACK!!)------------------------------------------------------
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_plane);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);

  seg.setDistanceThreshold (0.05);
  seg.setInputCloud (cloud_plane);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);

  float sum = 0;
  for (size_t i = 0; i < inliers_plane->indices.size (); ++i) {

    //cout << cloud_plane->points[inliers_plane->indices[i]].x << endl;
    sum = sum + cloud_plane->points[inliers_plane->indices[i]].x;
  }

  

  float mean_sum = sum/inliers_plane->indices.size ();

  //cout << "Sum: " << sum << endl;

  cout << "Mean_sum: " << mean_sum << endl;

  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  //transform_2.translation() << coefficients_plane->values[3], 0, 0;
  transform_2.translation() << -mean_sum, 0, 0;
  pcl::transformPointCloud(*cloud_b, *cloud_b, transform_2);

  pcl::transformPointCloud(*cloud_b, *cloud_b, transform);


  /*
  //-------------------------Create a conditional removal filter to remove brown pixels------------------------
  rMax = 255; //kept low to keep out orange from traffic cone
  rMin = 88; //numbers lower than 50 will yield green leaves
  gMax = 255; //kept low to keep out orange from traffic cone
  gMin = 58; //numbers lower than 50 will yield green leaves
  bMax = 255; //this blue is actually the critical value
  bMin = 37;
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond2 (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, rMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::GT, rMin)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, gMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::GT, gMin)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::LT, bMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, bMin)));

  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem2 (color_cond2);
  condrem2.setInputCloud (cloud_b);
  condrem2.setKeepOrganized(true);

  // apply filter
  condrem2.filter (*cloud_b);


  rMax = 255; //kept low to keep out orange from traffic cone
  rMin = 0; //numbers lower than 50 will yield green leaves
  gMax = 255; //kept low to keep out orange from traffic cone
  gMin = 0; //numbers lower than 50 will yield green leaves
  bMax = ; //this blue is actually the critical value
  bMin = 0;
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr color_cond2 (new pcl::ConditionAnd<pcl::PointXYZRGB> ());
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::LT, rMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("r", pcl::ComparisonOps::GT, rMin)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::LT, gMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("g", pcl::ComparisonOps::GT, gMin)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::LT, bMax)));
  color_cond2->addComparison (pcl::PackedRGBComparison<pcl::PointXYZRGB>::Ptr (new pcl::PackedRGBComparison<pcl::PointXYZRGB> ("b", pcl::ComparisonOps::GT, bMin)));

  // build the filter
  pcl::ConditionalRemoval<pcl::PointXYZRGB> condrem2 (color_cond2);
  condrem2.setInputCloud (cloud_b);
  condrem2.setKeepOrganized(true);

  // apply filter
  condrem2.filter (*cloud_b);
  */


  std::vector<int> indices2;
  pcl::removeNaNFromPointCloud(*cloud_b,*cloud_b, indices2);








  writer.write ("plants_tform.ply", *cloud_b, false);


  //Convert XYZRGB data to XYZ
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(*cloud_b, *cloud_xyz);

  /*

  // THIS IS WHERE GROW_UP STARTS
  //----------------------------------------Take Normals of tformed plants---------------------------------
  //(this might be redundant)
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::PointNormal>);
  pcl::search::Search<pcl::PointXYZ>::Ptr tree2;
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne2;
  tree2.reset (new pcl::search::KdTree<pcl::PointXYZ> (false));
  tree2->setInputCloud (cloud_xyz);

  ne2.setInputCloud (cloud_xyz);
  ne2.setSearchMethod (tree2);
  ne2.setRadiusSearch (.02);
  ne2.compute (*cloud_normals2);

  // OMG do I really have to hack this??
  for(int i=0; i<cloud_normals2->size(); i++){
    (*cloud_normals2)[i].x = (*cloud_xyz)[i].x;
    (*cloud_normals2)[i].y = (*cloud_xyz)[i].y;
    (*cloud_normals2)[i].z = (*cloud_xyz)[i].z;;
  }

  std::vector<int> indices2;
  pcl::removeNaNNormalsFromPointCloud(*cloud_normals2,*cloud_normals2, indices2);

//-----------------------------------------Take a 20cm Slice------------------------------------------
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_slice (new pcl::PointCloud<pcl::PointNormal>);
  // Create passthrough filter
  pcl::PassThrough<pcl::PointNormal> pass;
  pass.setInputCloud (cloud_normals2);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-0.2, -0.0);;
  pass.filter (*cloud_slice);
  cout << "Cloud_slice size: " << cloud_slice->size() << endl;
  //writer.write ("slice.ply", *cloud_slice, false);

  std::vector<int> indices3;
  pcl::removeNaNFromPointCloud(*cloud_slice,*cloud_slice, indices3);


  //-------------------------------------Heat Map / Filter It-----------------------------------

  cloud_slice = filterFunction2(cloud_slice);
  
  writer.write ("slice_filtered.ply", *cloud_slice, false);

  // Turn each cluster index into a point cloud
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> multiCloud;

  // Perform region growing on the sliced point cloud
  multiCloud = regionGrow_ground(cloud_slice);

  cout << "Number of clouds: " << multiCloud.size() << endl;

  //UNCOMMENT THIS TO WRITE THE GROUND-SLICE CLOUD CLUSTERS TO DISK

  for(int j=0; j<multiCloud.size(); j++){
    std::cout << "PointCloud representing the Cluster: " << (multiCloud[j])->points.size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".ply";
    writer.write<pcl::PointXYZ> (ss.str (), *(multiCloud[j]), false);
  }



  //----------------------------------THIS IS THE START OF GROWING UPWARDS--------------------------------

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_a_normals (new pcl::PointCloud<pcl::PointNormal>);
  std::map< int, std::map<int, std::vector<float> > > stalk_centroids2; // Create a map that will hold the stalk centroids

  int no_of_slices = 6;
  float extension = 0.035; //extended boundary of each upward growth
  int no_sections_threshold = 2; //there must be more than two recognized sections to be considered a stalk
  int min_stalk_points = 200;
  float slice_height = -0.20;


  for(int k=0; k<multiCloud.size(); k++) {

    std::stringstream ss3;
    ss3 << "cloud_cluster_" << k << ".ply";
      
    // Load one of the presumed stalks
    cloud1 = multiCloud[k];
    cout << "Size of imported cloud: " << cloud1->size() << endl;

    // Take the centroid of the first stalk section
    float x_sum=0; 
    float y_sum=0; 
    float z_sum=0;
    for (int j = 0; j < cloud1->size(); j++) {
      x_sum = (*cloud1)[j].x + x_sum;
      y_sum = (*cloud1)[j].y + y_sum;
      z_sum = (*cloud1)[j].z + z_sum;
    } 

    stalk_centroids2[k][0].push_back(x_sum/cloud1->size());
    stalk_centroids2[k][0].push_back(y_sum/cloud1->size());
    stalk_centroids2[k][0].push_back(z_sum/cloud1->size());

    for(int i=0; i<no_of_slices; i++) {
      pcl::PointXYZ minPt, maxPt;
      pcl::getMinMax3D (*cloud1, minPt, maxPt);

      //Select a particular section of the full cloud
      pcl::PointCloud<pcl::PointNormal>::Ptr cloud1_up(new pcl::PointCloud<pcl::PointNormal>);
      pcl::ConditionAnd<pcl::PointNormal>::Ptr yz_cond (new pcl::ConditionAnd<pcl::PointNormal> ());
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("y", pcl::ComparisonOps::GT, minPt.y-extension )));
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("y", pcl::ComparisonOps::LT, maxPt.y+extension )));
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("z", pcl::ComparisonOps::GT, minPt.z-extension )));
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("z", pcl::ComparisonOps::LT, maxPt.z+extension )));
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("x", pcl::ComparisonOps::GT, slice_height+slice_height*(i+1))));
      yz_cond->addComparison (pcl::FieldComparison<pcl::PointNormal>::ConstPtr (new pcl::FieldComparison<pcl::PointNormal> ("x", pcl::ComparisonOps::LT, slice_height*(i+1))));
      pcl::ConditionalRemoval<pcl::PointNormal> condrem;
      condrem.setCondition(yz_cond);
      condrem.setInputCloud(cloud_normals2);
      condrem.setKeepOrganized(true);
      condrem.filter (*cloud1_up);

      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloud1_up,*cloud1_up, indices);

      ////cout << "Number of points entering filterFunction: " << cloud1_up->size() << endl;

      cloud1_up = filterFunction(cloud1_up, maxPt, minPt, extension);

      std::stringstream ss;
      ss << "cloud" << k << "_" << i << "_up.ply";

      //Conclude whether this section is a stalk and add point location to a database
      if (cloud1_up->size() > min_stalk_points){
        cout << "Stalk!!  " << cloud1_up->size() << " points" << endl;
        ////writer.write (ss.str(), *cloud1_up, false);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_up_xyz(new pcl::PointCloud<pcl::PointXYZ>);

        //Save centroid of the stalk section and then clear everything
        x_sum=0; 
        y_sum=0; 
        z_sum=0;
        cloud1_up_xyz->points.resize(cloud1_up->size());
        for (int j = 0; j < cloud1_up->size(); j++) {
          (*cloud1_up_xyz)[j].x = (*cloud1_up)[j].x;
          (*cloud1_up_xyz)[j].y = (*cloud1_up)[j].y;
          (*cloud1_up_xyz)[j].z = (*cloud1_up)[j].z;
          x_sum = (*cloud1_up)[j].x + x_sum;
          y_sum = (*cloud1_up)[j].y + y_sum;
          z_sum = (*cloud1_up)[j].z + z_sum;
        }

        stalk_centroids2[k][i+1].push_back(x_sum/cloud1_up->size());
        stalk_centroids2[k][i+1].push_back(y_sum/cloud1_up->size());
        stalk_centroids2[k][i+1].push_back(z_sum/cloud1_up->size());

        cloud1 = cloud1_up_xyz;
        cloud1_up->clear();
        ////cout << "iteration: " << i << endl;
      }
      else {
        cout << "No Stalk!   " << cloud1_up->size() << " points" << endl;
        break;
      }
    }
  }

  // Assign the centroid as a point to the cloud called points_storage
  pcl::PointCloud<pcl::PointXYZ>::Ptr points_storage (new pcl::PointCloud<pcl::PointXYZ>); 
  points_storage->points.resize(no_of_slices+1);

  // Iterate through the two maps and store the centroids as cloud points
  int counter = 0;
  for(std::map<int,std::map<int,std::vector<float> > >::const_iterator ptr=stalk_centroids2.begin();ptr!=stalk_centroids2.end(); ptr++) {
      ////cout << ptr->first << "\n";
      for( std::map<int,std::vector<float> >::const_iterator eptr=ptr->second.begin();eptr!=ptr->second.end(); eptr++){
          ////cout << eptr->first << " " << eptr->second[0] << ", " << eptr->second[1] << ", " << eptr->second[2] << endl;
          (*points_storage)[eptr->first].x = eptr->second[0];
          (*points_storage)[eptr->first].y = eptr->second[1];
          (*points_storage)[eptr->first].z = eptr->second[2];
          counter++;
      }
      // Write a line between these points only if there are 3 or more sections found
      if(counter>no_sections_threshold){
        for(int w=0; w<(counter-1); w++){
          std::stringstream ss4;
        ss4 << "line_" << ptr->first << w;
          viewer->addLine<pcl::PointXYZ> ((*points_storage)[w], (*points_storage)[w+1], ss4.str());
        }
      }
      counter = 0;
  }


  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"Duration: "<< duration <<'\n';

  // Display the points
  //pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_a_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
  //reader.read<pcl::PointXYZRGB> ("plants_tform.ply", *cloud_a_rgb);

  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_b);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud_b, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

  */

  double feat_time = (clock() - start_global) / (double) CLOCKS_PER_SEC;
  cout << "TOTAL TIME: " << feat_time << endl;

  return 0;
}
