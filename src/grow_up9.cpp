#include <iostream>
#include <stdio.h>      /* printf */
#include <stdlib.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <math.h>
#include <sstream> 
#include <ctime>
#include <vector>


/*
This script loads the plants_tform.ply as 5 stitched point clouds. 
It grows the stalk location upward, looking for member elements.
*/

std::vector<std::vector<float> > filterFunction2(pcl::PointCloud<pcl::PointNormal>::Ptr cloud)
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
	std::vector<std::vector<float> > hotspots;
	std::vector<std::vector<float> > matrix_normals;
	std::map <std::string, std::vector<int> > point_indices; //this is a map that will be filled with a string of the matrix location and the associated point indices

	
	matrix.resize(no_of_rows, std::vector<int>(no_of_cols, initial_value));
	matrix_normals.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));
	hotspots.resize(50, std::vector<float>(2, initial_value));
	//queue.reserve(100, std::vector<int>(2, initial_value));

	// Loop through every point in the cloud, adding to the histogram	
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

	ofstream outFile;
	outFile.open("users.dat");

	float nanInsert = std::numeric_limits<float>::quiet_NaN();

	int noise_threshold = 30;
	int hotspot_threshold = 100;
	int I;
	int J;
	int counter = 0;
	//float Z;
	//float Y; 
	//cout << "No of rows: " << no_of_rows << endl;
	//cout << "No of cols: " << no_of_cols << endl;


	
	// This changes the histogram to be weighted by the normals as well
	for(int i=0; i<no_of_rows; i++)
	{
		for(int j=0; j<no_of_cols; j++) 
		{
			matrix_normals[i][j] = std::abs(matrix_normals[i][j])/(matrix[i][j]);
			matrix[i][j] = matrix[i][j]*(1-matrix_normals[i][j]);
		}
	}
	


	// THIS TAKES THE AVERAGE NORMAL VECTOR FOR EACH VOXEL AND THEN DECIDES TO KEEP OR TOSS THE ELEMENTS OF THE VOXEL
	for(int i=0; i<no_of_rows; i++)
	{
		for(int j=0; j<no_of_cols; j++) 
		{

			if(matrix[i][j]>hotspot_threshold){
				outFile << i << " " << j << " " << matrix[i][j] << "\n";
			}
			else if(matrix[i][j]>noise_threshold){
				outFile << i << " " << j << " " << matrix[i][j] << "\n";
			}
			else{
				outFile << i << " " << j << " " << 0 << "\n";				
			}

			if ((matrix[i][j]>hotspot_threshold)) { //&& (matrix_normals[i][j])<0.5) {

				std::vector<std::vector<int> > queue;
				
				// Add point to hotspots vector
				hotspots[counter][0] = minPt.z + float(j)/100;
				hotspots[counter][1] = minPt.y + float(i)/100;

				//cout << "Hotspot density: " << matrix[i][j] << endl;

				// Place hotspot in queue
				std::vector<int> myvector;
  				myvector.push_back(i);
  				myvector.push_back(j);  
  				//cout << "i: " << i << ", j: " << j << endl;			
    			queue.push_back(myvector);
    			myvector.clear();

    			int new_i;
    			int new_j;
    			int whilecounter = 0;

    			// Start a while-loop clearing elements in a 5x5 grid around the queue elements
    			//cout << "Queue size: " << queue.size() << endl;
    			while(queue.size()){
    			//for(int z=0; z<2; z++){
					for(int k=-2; k<3; k++){
						for(int l=-2; l<3; l++){
							new_i = queue[0][0];
							new_j = queue[0][1];
							//cout << "new_i: " << new_i << ", new_j: " << new_j << endl;
							if((k+new_i<no_of_rows)&&(k+new_i>=0)&&(l+new_j<no_of_cols)&&(l+new_j>=0)) {
								if(((k==-2)||(k==2)||(l==-2)||(l==2))&&(matrix[k+new_i][l+new_j]>noise_threshold)){  //RIGHT HERE IS THE NOISE THRESHOLD!!
									myvector.push_back(new_i+k); 
									myvector.push_back(new_j+l);  				
					    			queue.push_back(myvector);
					    			myvector.clear();
					    			//cout << "HELLOOOOOO" << endl;
								}
								matrix[k+new_i][l+new_j] = 0; // Decimate the spot
								whilecounter++;
								//cout << "Queue size: " << queue.size() << endl;
							}
						}
					}
					//queue.pop_front();
					queue.erase(queue.begin()); 
					//cout << "Queue size: " << queue.size() << endl;
    			}

    			//cout << "Queue size: " << queue.size() << endl;

				counter++;
				cout << "Whilecounter is: " << whilecounter << endl;

			}

			//else{
			//	outFile << i << " " << j << " " << 0 << "\n";				
			//}
		}
	}

    outFile.close();

    //cout << "AND HERE" << endl;

    //cout << hotspots[0][0] << " " << hotspots[0][1] << endl;

 	return(hotspots);
}


int main (int argc, char** argv)
{

    pcl::PLYReader reader;
    pcl::PLYWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a (new pcl::PointCloud<pcl::PointXYZ>);  

    // Import just the regular unmodified (other than rotated) point cloud
    reader.read<pcl::PointXYZ> ("stitched_clouds2/plants_tform_101.ply", *cloud_a);
    cout << "Cloud_a size: " << cloud_a->size() << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    pcl::PassThrough<pcl::PointXYZ> pass_new;
    pass_new.setInputCloud (cloud_a);
    pass_new.setFilterFieldName ("z");
    pass_new.setFilterLimits (-1.6, -1.2);
    pass_new.filter (*cloud_a);

    std::clock_t start;
    double duration;
    start = std::clock();

    //----------------------------------------Take Normals of tformed plants---------------------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
	pcl::search::Search<pcl::PointXYZ>::Ptr tree;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
	tree.reset (new pcl::search::KdTree<pcl::PointXYZ> (false));
	tree->setInputCloud (cloud_a);

	//writer.write ("cloud_a.ply", *cloud_a, false);

	ne.setInputCloud (cloud_a);
	ne.setSearchMethod (tree);
	ne.setRadiusSearch (.02);
	ne.compute (*cloud_normals);

	// OMG do I really have to hack this??
	for(int i=0; i<cloud_normals->size(); i++){
		(*cloud_normals)[i].x = (*cloud_a)[i].x;
		(*cloud_normals)[i].y = (*cloud_a)[i].y;
		(*cloud_normals)[i].z = (*cloud_a)[i].z;;
	}

	std::vector<int> indices;
    pcl::removeNaNNormalsFromPointCloud(*cloud_normals,*cloud_normals, indices);

	//std::cerr << ">> Done: " << cloud_normals->size() << "\n";
	//writer.write ("cloud_normals.ply", *cloud_normals, false);

	// Turn each cluster index into a point cloud
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> multiCloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr spheres (new pcl::PointCloud<pcl::PointXYZ>);

    for(int n=0; n<8; n++){
	    //-----------------------------------------Take a 20cm Slice------------------------------------------
	    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_slice (new pcl::PointCloud<pcl::PointNormal>);

	    // Create passthrough filter and slice the cloud
	    pcl::PassThrough<pcl::PointNormal> pass;
	    pass.setInputCloud (cloud_normals);
	    pass.setFilterFieldName ("x");
	    pass.setFilterLimits (-0.2-0.1*(n), -0.1*(n));
	    //pass.setFilterLimits (-0.2, -0.0);
	    pass.filter (*cloud_slice);
	    cout << "Cloud_slice size: " << cloud_slice->size() << endl;
	    writer.write ("slice.ply", *cloud_slice, false);

		std::vector<int> indices3;
	    pcl::removeNaNFromPointCloud(*cloud_slice,*cloud_slice, indices3);


	    //-------------------------------------Heat Map / Filter It-----------------------------------

		std::vector<std::vector<float> > hotspots;
		//hotspots.resize(500, std::vector<float>(2, 0));
	    // filter one slice of the cloud
	    hotspots = filterFunction2(cloud_slice);    

	    //cout << "Z: " << hotspots[0][0] << ", Y: " << hotspots[0][1] << endl;
	    //float Z = hotspots[0][0];
	    //float Y = hotspots[0][1];
		
		//pcl::PointCloud<pcl::PointXYZ>::Ptr spheres (new pcl::PointCloud<pcl::PointXYZ>);
		//cout << hotspots.size() << endl;

		spheres->width    = hotspots.size();
		spheres->height   = 1;
		spheres->is_dense = false;
		spheres->points.resize (spheres->width * spheres->height);

		for(int i=0; i<hotspots.size(); i++){
			if(hotspots[i][0]!=0){
				//cout << "i is: " << i << endl;
			    spheres->points[i].x = -0.1-0.1*(n);
			    spheres->points[i].y = hotspots[i][1];
			    spheres->points[i].z = hotspots[i][0];
			}
		}

    	std::stringstream ss;
	  	for(int i=0; i<spheres->points.size(); i++){
	  		if(spheres->points[i].z != 0) {
	  			ss << "sphere" << i << "_" << n;
	  			viewer->addSphere (spheres->points[i], 0.025, 1., 0.0, 0.0, ss.str()); 
	  		} 		
	  	}

		// write just the filtered slice (no region growing yet)
	    writer.write ("slice_filtered.ply", *cloud_slice, false);
	}



	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Duration: "<< duration <<'\n';

	// Display the points
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_a_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read<pcl::PointXYZRGB> ("stitched_clouds2/plants_tform_101.ply", *cloud_a_rgb);
	pcl::PassThrough<pcl::PointXYZRGB> pass_new2;
    pass_new2.setInputCloud (cloud_a_rgb);
    pass_new2.setFilterFieldName ("z");
    pass_new2.setFilterLimits (-1.6, -1.2);
    pass_new2.filter (*cloud_a_rgb);



	viewer->setBackgroundColor (0, 0, 0);
  	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_a_rgb);
  	viewer->addPointCloud<pcl::PointXYZRGB> (cloud_a_rgb, rgb, "sample cloud");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");



	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}

	
    return(0);

}