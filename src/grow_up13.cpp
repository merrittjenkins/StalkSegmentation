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
#include <Eigen/Core>
#include <algorithm>
#include <iterator>
#include "Vec3.h"


/*
This script loads the plants_tform.ply as 5 stitched point clouds. 
It grows the stalk location upward, looking for member elements.
*/

#include <pcl/point_cloud.h>

struct MyPointType
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  int layer;
  float weight;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (MyPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, layer, layer)
                                   (float, weight, weight)
)



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
	hotspots.resize(50, std::vector<float>(3, initial_value));
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

	int noise_threshold = 40;
	int hotspot_threshold = 80;
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
				outFile << i << " " << j << " " << matrix[i][j] << "\n"; // change this vale to 300 if you want to see the hotspots
			}
			else if(matrix[i][j]>noise_threshold){
				outFile << i << " " << j << " " << matrix[i][j] << "\n";
			}
			else{
				outFile << i << " " << j << " " << 0 << "\n";				
			}

			// Comment out this if-statement if using GNUPLOT
			
			if ((matrix[i][j]>hotspot_threshold)) { //&& (matrix_normals[i][j])<0.5) {

				std::vector<std::vector<int> > queue;
				
				// Add point to hotspots vector
				hotspots[counter][0] = minPt.z + float(j)/100;
				hotspots[counter][1] = minPt.y + float(i)/100;
				hotspots[counter][2] = matrix[i][j];

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
    			int loc_y=0;
    			int loc_z=0;
    			int whilecounter = 0;
    			float temp = matrix[i][j];

    			// Start a while-loop clearing elements in a 5x5 grid around the queue elements
    			//cout << "Queue size: " << queue.size() << endl;
    			while(queue.size()){
    			//for(int z=0; z<2; z++){
					for(int k=-1; k<2; k++){
						for(int l=-1; l<2; l++){
							new_i = queue[0][0];
							new_j = queue[0][1];
							//cout << "new_i: " << new_i << ", new_j: " << new_j << endl;
							if((k+new_i<no_of_rows)&&(k+new_i>=0)&&(l+new_j<no_of_cols)&&(l+new_j>=0)) {  // Check to make sure the square is within the boundary
								if(((k==-1)||(k==1)||(l==-1)||(l==1))&&(matrix[k+new_i][l+new_j]>noise_threshold)){  //RIGHT HERE IS THE NOISE THRESHOLD!!
									myvector.push_back(new_i+k); 
									myvector.push_back(new_j+l);  				
					    			queue.push_back(myvector);
					    			myvector.clear();
					    			//cout << "HELLOOOOOO" << endl;
								}

								// Save the largest value found in the region search
								if(matrix[k+new_i][l+new_j] > temp){
									temp = matrix[k+new_i][l+new_j];
									//cout << "temp: " << temp << endl;
									hotspots[counter][0] = minPt.z + float(l+new_j)/100;
									hotspots[counter][1] = minPt.y + float(k+new_i)/100;
									hotspots[counter][2] = matrix[k+new_i][l+new_j];
								}

								//if(matrix[k+new_i][l+new_j]>0)
									//cout << "matrix value: " << matrix[k+new_i][l+new_j] << endl;

								// Decimate the spot
								if (matrix[k+new_i][l+new_j] > noise_threshold){
									loc_y+=(k+new_i);
									loc_z+=(l+new_j);
									matrix[k+new_i][l+new_j] = 0;
									whilecounter++; 
								}
								//cout << "Queue size: " << queue.size() << endl;
							}
						}
					}
					//loc_y = loc_y/whilecounter;
					//loc_z = loc_z/whilecounter;
					//hotspots[counter][0] = minPt.z + float(loc_z)/100;
					//hotspots[counter][1] = minPt.y + float(loc_y)/100;
					//queue.pop_front();
					queue.erase(queue.begin()); 
					//cout << "Queue size: " << queue.size() << endl;
    			}

    			//cout << "Hotspot y-loc: " << hotspots[counter][1] <<", loc_y: " << minPt.y + float(loc_y)/100 << endl;
    			//cout << "loc_y: " << loc_y << ", whilecounter: " << whilecounter << ", division: " << loc_y/whilecounter << ", i: " << i << endl;
    			hotspots[counter][1] = minPt.y + float(loc_y/whilecounter)/100;
    			hotspots[counter][0] = minPt.z + float(loc_z/whilecounter)/100;

    			//cout << "Queue size: " << queue.size() << endl;

				counter++;
				//cout << "Whilecounter is: " << whilecounter << endl;

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




std::vector<std::vector<float> > sphereLogic(pcl::PointCloud<MyPointType>::Ptr sphereCloud){
    //pcl::PLYWriter writer;
    //writer.write ("sphere_locs.ply", *sphereCloud, false);

	std::vector<std::vector<float> > sphere_locs;
	sphere_locs.resize(50, std::vector<float>(6, 0));

	return(sphere_locs);
}

void display_vector(const std::vector<float> &v)
{
    std::copy(v.begin(), v.end(),
        std::ostream_iterator<float>(std::cout, " "));
}


std::vector<std::vector<float> > sphereLogic2(std::vector<std::vector<std::vector<float> > > sphere_locs_3d){

	int sphere_memory = 50;

	std::vector<std::vector<float> > sphere_locs (0,std::vector<float>(8, 0));
	//sphere_locs.resize(sphere_memory, std::vector<float>(6, 0));
	std::vector<std::vector<float> > RANSAC_vec (0,std::vector<float>(3, 0));
	std::vector<std::vector<float> > inliers_vec (0,std::vector<float>(3, 0));
	std::vector<std::vector<float> > inliers_temp_vec (0,std::vector<float>(3, 0));
	std::vector<float> distfinal_vec (0, 0);
	std::vector<float> distfinal_temp_vec (0, 0);

	//FILL THE SPHERE_LOCS VECTOR WITH CYLINDER LOCATION PAIRS
	int num_slices = sphere_locs_3d.size();
	int counter = 0;
	float z_point; float y_point; float x_point;
	float z_threshold = 0.02; //0.02
	float y_threshold = 0.02; //0.02
	float RANSAC_threshold = 0.025;

	//iterate through each point in the first layer
	for(int i=0; i<sphere_locs_3d[0].size(); i++){ 
		if (sphere_locs_3d[0][i][2]!=0){
			x_point = sphere_locs_3d[0][i][0];
			y_point = sphere_locs_3d[0][i][1];
			z_point = sphere_locs_3d[0][i][2];

			// Search for points in other layers within a z-y boundary
			for(int slice=1; slice<(num_slices); slice++){ //iterate through each slice other than the first
				for(int j=0; j<sphere_locs_3d[slice].size(); j++){ //iterate through each point
					if ((std::abs(z_point - sphere_locs_3d[slice][j][2]) < slice*z_threshold)&&(std::abs(y_point - sphere_locs_3d[slice][j][1]) < slice*y_threshold)){ //z and y threshold on connected stalks
						
						//Write the points to a vector; 
						RANSAC_vec.push_back(sphere_locs_3d[slice][j]);  //RANSAC_vec is all of the inliers related to a particular base point
					}
				}
			}

			int inliers_temp = 0;
			float dist_sum = 0;
			Vec3<double> C(x_point, y_point, z_point); // Define the base point as a Vec3
			std::vector<float> optimal; //(0.,0.,0.);
			optimal.resize(3,0);

			// Loop through all points in RANSAC_vec, each defining a line
			for(int iterator=0; iterator<RANSAC_vec.size(); iterator++){

				//Define the line between C and B
	    		Vec3<double> B(RANSAC_vec[iterator][0], RANSAC_vec[iterator][1], RANSAC_vec[iterator][2]); // this is the other point defining the line
				int inliers_counter = 0;

				for(int j=0; j<RANSAC_vec.size(); j++){ //Loop through all points in RANSAC_vec, checking the distance from the line

					//Check the distance from the line for each point
					Vec3<double> A (RANSAC_vec[j][0], RANSAC_vec[j][1], RANSAC_vec[j][2]);
					double denominator = pow(Vec3<double>::getDistance(C, B),2);
					double numerator = Vec3<double>::dotProduct((A-C), (B-C));
					Vec3<double> dist_vec = (A-C)-(B-C)*(numerator/denominator);
					double dist_final = sqrt(Vec3<double>::dotProduct(dist_vec, dist_vec));
				    if(dist_final < RANSAC_threshold){
				    	// save inliers here in a vector
				    	std::vector<float> myvector;
		  				myvector.push_back(RANSAC_vec[j][0]);
		  				myvector.push_back(RANSAC_vec[j][1]);
		  				myvector.push_back(RANSAC_vec[j][2]);
						inliers_temp_vec.push_back(myvector);
						distfinal_temp_vec.push_back(dist_final);
						//cout << "Dist final: " << dist_final << endl;
						//cout << RANSAC_vec[j][0] << endl;
						myvector.clear();

				    	inliers_counter++;
				    }
				}
				if(inliers_counter>=inliers_temp){
					inliers_temp = inliers_counter;
					optimal[0] = RANSAC_vec[iterator][0];
					optimal[1] = RANSAC_vec[iterator][1];
					optimal[2] = RANSAC_vec[iterator][2];
					//dist_sum = std::accumulate(distfinal_temp_vec.begin(), distfinal_temp_vec.end(), 0);
					//cout << "DIST SUM IS: " << dist_sum << endl;
					//append vector of inliers
					for(int i=0; i<inliers_temp_vec.size(); i++){
						inliers_vec.push_back(inliers_temp_vec[i]);
						dist_sum+=distfinal_temp_vec[i];
					}
					//cout << "DIST SUM IS: " << dist_sum << endl;
				}
				inliers_temp_vec.clear();
				distfinal_temp_vec.clear();
			}

			// Add the Slice0 point and the optimal point to sphere_locs
			if(inliers_temp>2){ //3
				//int num_inliers = 0;
				std::vector<float> myvector;
  				myvector.push_back(x_point);
  				myvector.push_back(y_point);
  				myvector.push_back(z_point);
  				myvector.push_back(optimal[0]); 
  				myvector.push_back(optimal[1]); 
  				myvector.push_back(optimal[2]); 
  				myvector.push_back(inliers_temp); 
  				myvector.push_back(dist_sum); 
				sphere_locs.push_back(myvector);
				myvector.clear();
			}
			//display_vector(optimal);
			optimal.clear();
			RANSAC_vec.clear();
			counter++;
		}
	}

	//Remove spheres captured from the first row from sphere_locs_3d
	for(int i=0; i<inliers_vec.size(); i++){
		//cout << "X: " << inliers_vec[i][0] << ", Y:" << inliers_vec[i][1] << ", Z:" << inliers_vec[i][2] << endl;
		for(int layer=0; layer<sphere_locs_3d.size(); layer++){ //iterate through the six layers
			for(int j=0; j<sphere_locs_3d.size(); j++){ //iterate through the spheres in each layer			
				if((inliers_vec[i][0]==sphere_locs_3d[layer][j][0])&&(inliers_vec[i][1]==sphere_locs_3d[layer][j][1])&&(inliers_vec[i][2]==sphere_locs_3d[layer][j][2])){
					//cout << "in the first" << endl;
					cout << "X: " << inliers_vec[i][0] << ", Y:" << inliers_vec[i][1] << ", Z:" << inliers_vec[i][2] << endl;
					sphere_locs_3d[layer][j][0] = 0;
					sphere_locs_3d[layer][j][1] = 0;
					sphere_locs_3d[layer][j][2] = 0;
				}
			}
		}
	}

	/*
	for(int i=0; i<sphere_locs.size(); i++){
		for(int layer=0; layer<sphere_locs_3d.size(); layer++){ //iterate through the six layers
			for(int j=0; j<sphere_locs_3d.size(); j++){ //iterate through the spheres in each layer			
				if((sphere_locs[i][0]==sphere_locs_3d[layer][j][0])&&(sphere_locs[i][1]==sphere_locs_3d[layer][j][1])&&(sphere_locs[i][2]==sphere_locs_3d[layer][j][2])){
					cout << "in the first" << endl;
					sphere_locs_3d[layer][j][0] = 0;
					sphere_locs_3d[layer][j][1] = 0;
					sphere_locs_3d[layer][j][2] = 0;
				}
				if((sphere_locs[i][3]==sphere_locs_3d[layer][j][0])&&(sphere_locs[i][4]==sphere_locs_3d[layer][j][1])&&(sphere_locs[i][5]==sphere_locs_3d[layer][j][2])){
					cout << "in the second" << endl;
					sphere_locs_3d[layer][j][0] = 0;
					sphere_locs_3d[layer][j][1] = 0;
					sphere_locs_3d[layer][j][2] = 0;
				}
			}
		}
	}
	*/

	float dist_sum = 0;
	for(int i=0; i<sphere_locs_3d[1].size(); i++){ 
		if (sphere_locs_3d[1][i][2]!=0){
			x_point = sphere_locs_3d[1][i][0];
			y_point = sphere_locs_3d[1][i][1];
			z_point = sphere_locs_3d[1][i][2];

			// Search for points in other layers within a z-y boundary
			for(int slice=2; slice<(num_slices); slice++){ //iterate through each slice other than the first
				for(int j=0; j<sphere_locs_3d[slice].size(); j++){ //iterate through each point
					if ((std::abs(z_point - sphere_locs_3d[slice][j][2]) < slice*z_threshold)&&(std::abs(y_point - sphere_locs_3d[slice][j][1]) < slice*y_threshold)){ //z and y threshold on connected stalks
						
						//Write the points to a vector; 
						RANSAC_vec.push_back(sphere_locs_3d[slice][j]);  //RANSAC_vec is all of the inliers related to a particular base point
					}
				}
			}

			int inliers_temp = 0;
			Vec3<double> C(x_point, y_point, z_point); // Define the base point as a Vec3
			std::vector<float> optimal; //(0.,0.,0.);
			optimal.resize(3,0);

			// Loop through all points in RANSAC_vec, each defining a line
			for(int iterator=0; iterator<RANSAC_vec.size(); iterator++){

				//Define the line between C and B
	    		Vec3<double> B(RANSAC_vec[iterator][0], RANSAC_vec[iterator][1], RANSAC_vec[iterator][2]); // this is the other point defining the line
				int inliers_counter = 0;

				for(int j=0; j<RANSAC_vec.size(); j++){ //Loop through all points in RANSAC_vec, checking the distance from the line

					//Check the distance from the line for each point
					Vec3<double> A (RANSAC_vec[j][0], RANSAC_vec[j][1], RANSAC_vec[j][2]);
					double denominator = pow(Vec3<double>::getDistance(C, B),2);
					double numerator = Vec3<double>::dotProduct((A-C), (B-C));
					Vec3<double> dist_vec = (A-C)-(B-C)*(numerator/denominator);
					double dist_final = sqrt(Vec3<double>::dotProduct(dist_vec, dist_vec));
				    //cout << "dist_final: " << dist_final << ", ";
				    if(dist_final < RANSAC_threshold){
				    	//cout << "X-loc: " << RANSAC_vec[j][0] << ", Y-loc: " << RANSAC_vec[j][1] << ", Z-loc: " << RANSAC_vec[j][2] << endl;
				    	distfinal_temp_vec.push_back(dist_final);
				    	inliers_counter++;
				    }
				    //cout << "inliers counter is: " << inliers_counter << endl;
				}
				//cout << "Endfor" << endl;
				if(inliers_counter>=inliers_temp){
					inliers_temp = inliers_counter;
					optimal[0] = RANSAC_vec[iterator][0];
					optimal[1] = RANSAC_vec[iterator][1];
					optimal[2] = RANSAC_vec[iterator][2];
					for(int i=0; i<distfinal_temp_vec.size(); i++){
						dist_sum+=distfinal_temp_vec[i];
					}
				}
				distfinal_temp_vec.clear();
			}
			//cout << "inliers_temp: " << inliers_temp << endl;
			// Add the Slice0 point and the optimal point to sphere_locs
			if(inliers_temp>2){ //3
				//int num_inliers=0;
				std::vector<float> myvector;
  				myvector.push_back(x_point);
  				myvector.push_back(y_point);
  				myvector.push_back(z_point);
  				myvector.push_back(optimal[0]); 
  				myvector.push_back(optimal[1]); 
  				myvector.push_back(optimal[2]); 
  				myvector.push_back(inliers_temp); 
  				myvector.push_back(dist_sum); 
				sphere_locs.push_back(myvector);
				myvector.clear();
			}
			//display_vector(optimal);
			optimal.clear();
			RANSAC_vec.clear();
			counter++;
		}
	}
    
 

	return(sphere_locs);
}


/*
std::vector<std::vector<float> > sphereLogic2(std::vector<std::vector<std::vector<float> > > sphere_locs_3d){

	int sphere_memory = 50;

	std::vector<std::vector<float> > sphere_locs;
	sphere_locs.resize(sphere_memory, std::vector<float>(6, 0));

	//FILL THE SPHERE_LOCS VECTOR WITH CYLINDER LOCATION PAIRS
	int num_slices = sphere_locs_3d.size();
	int counter = 0;
	float z_point; float y_point; float x_point;
	for(int i=0; i<(num_slices-1); i++){ //iterate through each slice
		for(int j=0; j<sphere_locs_3d[i].size(); j++){ //iterate through each point
			if(sphere_locs_3d[i][j][2]!=0){ //if the z-location isn't 1 (ie if the vector is filled)
				//cout << sphere_locs_3d[i][j][2] << endl;
				x_point = sphere_locs_3d[i][j][0];
				y_point = sphere_locs_3d[i][j][1];
				z_point = sphere_locs_3d[i][j][2];
				for(int k=0; k<sphere_locs_3d[i+1].size(); k++){ //iterate through each point of the next slice up
					if ((std::abs(z_point - sphere_locs_3d[i+1][k][2]) < 0.05)&&(std::abs(y_point - sphere_locs_3d[i+1][k][1]) < 0.05)){ //z and y threshold on connected stalks
						sphere_locs[counter][0] = x_point;
						sphere_locs[counter][1] = y_point;
						sphere_locs[counter][2] = z_point;
						sphere_locs[counter][3] = sphere_locs_3d[i+1][k][0];
						sphere_locs[counter][4] = sphere_locs_3d[i+1][k][1];
						sphere_locs[counter][5] = sphere_locs_3d[i+1][k][2];
						counter++;
					}
				}
			}
		}
		//cout << "Out of inner for" << endl;
		//counter = 0;
	}

	cout << "Size of sphere_locs_3d.size(): " << sphere_locs_3d.size() << endl;

	//Remove cylinders defined by one point pair
	
	//int flag = 0;
	//for(int i=0; i<sphere_memory; i++){
	//	for(int j=0; j<sphere_memory; j++){
	//		if((sphere_locs[i][3] == sphere_locs[j][0]) && (sphere_locs[i][4] == sphere_locs[j][1]) && (sphere_locs[i][5] == sphere_locs[j][2])){
	//			flag = 1;	
	//		}
	//	}
	//	if(flag==0){
	//		sphere_locs.erase(sphere_locs.begin()+i);
	//	}
	//	flag = 0;
	//}
	

	return(sphere_locs);
}
*/


// dist3D_Segment_to_Segment(): get the 3D minimum distance between 2 segments
//    Input:  two 3D line segments S1 and S2
//    Return: the shortest distance between S1 and S2
float dist3D_Segment_to_Segment(Vec3<double> P1_0, Vec3<double> P1_1, Vec3<double> P2_0, Vec3<double> P2_1)
{
	float SMALL_NUM = 0.00000001;
    //Vector   u = S1.P1 - S1.P0;
    Vec3<double> u = P1_1 - P1_0;
    //Vector   v = S2.P1 - S2.P0;
    Vec3<double> v = P2_1 - P2_0;
    //Vector   w = S1.P0 - S2.P0;
    Vec3<double> w = P1_0 - P2_0;
    //float    a = dot(u,u);         // always >= 0
    float a = Vec3<double>::dotProduct(u,u);
    //float    b = dot(u,v);
    float b = Vec3<double>::dotProduct(u,v);
    //float    c = dot(v,v);         // always >= 0
    float c = Vec3<double>::dotProduct(v,v);
    //float    d = dot(u,w);
    float d = Vec3<double>::dotProduct(u,w);
    //float    e = dot(v,w);
    float e = Vec3<double>::dotProduct(v,w);
    float    D = a*c - b*b;        // always >= 0
    float    sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    float    tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    
    // compute the line parameters of the two closest points
    if (D < SMALL_NUM) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d +  b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD);
    tc = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD);

    
    // get the difference of the two closest points
    Vec3<double> dP = w + (u*sc) - (v*tc);  // =  S1(sc) - S2(tc)
    
    float dwP = sqrt(Vec3<double>::dotProduct(dP,dP));

    return (dwP);//norm(dP);   // return the closest distance
}





int main (int argc, char** argv)
{

    pcl::PLYReader reader;
    pcl::PLYWriter writer;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a (new pcl::PointCloud<pcl::PointXYZ>);  

    cout << "Argv is: " << argv[1] << endl;

    // Import just the regular unmodified (other than rotated) point cloud
    std::stringstream ss5;
    ss5 << "stitched_clouds2_complete/plants_tform_" << argv[1] << ".ply";
    //reader.read<pcl::PointXYZ> ("stitched_clouds2/plants_tform_111.ply", *cloud_a);
    reader.read<pcl::PointXYZ> (ss5.str(), *cloud_a);
    cout << "Cloud_a size: " << cloud_a->size() << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

    pcl::PassThrough<pcl::PointXYZ> pass_new;
    pass_new.setInputCloud (cloud_a);
    pass_new.setFilterFieldName ("z");
    pass_new.setFilterLimits (-1.6, -1.25);
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
	//std::vector<std::vector<float> > sphere_locs;
	int num_slices = 6;
	std::vector<std::vector<std::vector<float> > > sphere_locs_3d (num_slices,std::vector<std::vector<float> >(15,std::vector<float>(3, 0)));  // HARD CODING 15 SPHERES IN ONE LAYER!!!
 	//sphere_locs.resize(50, std::vector<float>(2, 0));

	// Turn each cluster index into a point cloud
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> multiCloud;

	pcl::PointCloud<MyPointType>::Ptr sphereCloud (new pcl::PointCloud<MyPointType>);
  	sphereCloud->width = 100;
  	sphereCloud->height = 1;
  	sphereCloud->is_dense = false;
	sphereCloud->points.resize (sphereCloud->width * sphereCloud->height);

  	int globalSphereCount = 0;

    for(int n=0; n<num_slices; n++){
	    //-----------------------------------------Take a 20cm Slice------------------------------------------
	    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_slice (new pcl::PointCloud<pcl::PointNormal>);

	    // Create passthrough filter and slice the cloud
	    pcl::PassThrough<pcl::PointNormal> pass;
	    pass.setInputCloud (cloud_normals);
	    pass.setFilterFieldName ("x");
	    pass.setFilterLimits (-0.2-0.1*(n), -0.1*(n));
	    //pass.setFilterLimits (-0.2, -0.0);
	    pass.filter (*cloud_slice);
	    //cout << "Cloud_slice size: " << cloud_slice->size() << endl;
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

	    pcl::PointCloud<pcl::PointXYZ>::Ptr spheres (new pcl::PointCloud<pcl::PointXYZ>);
		spheres->width    = hotspots.size(); //this is 50 because I assign the vector hotspots to be arbitrarily large
		spheres->height   = 1;
		spheres->is_dense = false;
		spheres->points.resize (spheres->width * spheres->height);

		
		for(int i=0; i<hotspots.size(); i++){
			if(hotspots[i][0]!=0){
			    spheres->points[i].x = -0.1-0.1*(n);
			    spheres->points[i].y = hotspots[i][1];
			    spheres->points[i].z = hotspots[i][0];
			    sphereCloud->points[globalSphereCount].x = -0.1-0.1*(n);
			    sphereCloud->points[globalSphereCount].y = hotspots[i][1];
			    sphereCloud->points[globalSphereCount].z = hotspots[i][0];
			    sphereCloud->points[globalSphereCount].layer = n;
			    sphereCloud->points[globalSphereCount].weight = 0;
			    sphere_locs_3d[n][i][0] = -0.1-0.1*(n);
	  			sphere_locs_3d[n][i][1] = hotspots[i][1];
	  			sphere_locs_3d[n][i][2] = hotspots[i][0];
			    globalSphereCount++;
			}
		}
		

    	std::stringstream ss;
    	int sphereCount = 0;
	  	for(int i=0; i<spheres->points.size(); i++){ //there are also 50 spheres becayse 
	  		if(spheres->points[i].z != 0) {
	  			//cout << "n: " << n << ", i: " << i << ", sphereCount: " << sphereCount << endl;
	  			ss << "sphere" << i << "_" << n;

	  			//REINSTATE IF I WANT SPHERE WEIGHTS INCLUDED IN COLOR
	  			//viewer->addSphere (spheres->points[i], 0.025, (hotspots[sphereCount][2]-100)/100, (hotspots[sphereCount][2]-100)/100, 0.0, ss.str()); 
	  			viewer->addSphere (spheres->points[i], 0.025, 1.0, 1.0, 0.0, ss.str()); 
	  			//cout << "Color: " << (hotspots[sphereCount][2]-100)/100 << endl;
	  			sphereCount++;
	  		} 		
	  	}
	  	//cout << "sphereCount is: " << sphereCount << endl;
	  	

		// write just the filtered slice (no region growing yet)
	    writer.write ("slice_filtered.ply", *cloud_slice, false);
	}

    	
	std::vector<std::vector<float> > sphere_locs;
	//sphere_locs = sphereLogic(sphereCloud);
	sphere_locs = sphereLogic2(sphere_locs_3d);


	float dist;
	float cylinder_dist_threshold = 0.11;
	std::vector<int> to_remove;
	//to_remove.resize(0,0);
	cout << "Number of cylinders: " << sphere_locs.size() << endl; 

	sphere_locs[0][5] = sphere_locs[0][5]+0.01;

	//Check if any cylinders are too close together
	
	for(int i=0; i<sphere_locs.size(); i++){
		cout << sphere_locs[i][0] << " " << sphere_locs[i][1] << " " << sphere_locs[i][2] << " ";
		cout << sphere_locs[i][3] << " " << sphere_locs[i][4] << " " << sphere_locs[i][5] << endl;
		for(int j=0; j<sphere_locs.size(); j++){
			if(i!=j){
				Vec3<double> P1_0 (sphere_locs[i][0], sphere_locs[i][1], sphere_locs[i][2]);
				Vec3<double> P1_1 (sphere_locs[i][3], sphere_locs[i][4], sphere_locs[i][5]);
				Vec3<double> P2_0 (sphere_locs[j][0], sphere_locs[j][1], sphere_locs[j][2]);
				Vec3<double> P2_1 (sphere_locs[j][3], sphere_locs[j][4], sphere_locs[j][5]);				
				dist = dist3D_Segment_to_Segment(P1_0, P1_1, P2_0, P2_1);
				//cout << "Distance: " << dist << ", i: " << i << ", j:" << j << endl;

				//BIG GOOFY HACK HERE!!!
				if((sphere_locs[i][3] == sphere_locs[j][3])&&(sphere_locs[i][4]==sphere_locs[j][4])&&(sphere_locs[i][5]==sphere_locs[j][5])){ //if they share an upper point...remove
					dist = 0;
				}

				if(dist < cylinder_dist_threshold){
					if(sphere_locs[i][6]>sphere_locs[j][6]){ // check if the distance is below the threshold and compare number of points captured
						to_remove.push_back(j);
						//cout << "Weight: " << sphere_locs[i][6] << ", Weight: " << sphere_locs[j][6] << endl;
					}
					else if(sphere_locs[i][6]<sphere_locs[j][6]) {
						to_remove.push_back(i);
					}
					else{
						cout << "------------THEY'RE EQUAL!------------" << endl;
						cout << "dist_sum: " << sphere_locs[i][7] << ", " << sphere_locs[j][7] << endl;
						if(sphere_locs[i][7]>sphere_locs[j][7]){ //if point i has a larger error than point j
							to_remove.push_back(i);
						}
						else{
							to_remove.push_back(j);
						}
					}
				}
			}
		}	
	}
	

	//cout << "to remove: " << to_remove << endl;
	for(int i=0; i<to_remove.size(); i++){
		cout << "To REMOVE: " << to_remove[i] << endl;
	}



	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Duration: "<< duration <<'\n';

	// Display the points
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_a_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read<pcl::PointXYZRGB> (ss5.str(), *cloud_a_rgb);
	pcl::PassThrough<pcl::PointXYZRGB> pass_new2;
    pass_new2.setInputCloud (cloud_a_rgb);
    pass_new2.setFilterFieldName ("z");
    pass_new2.setFilterLimits (-1.6, -1.25);
    pass_new2.filter (*cloud_a_rgb);



	viewer->setBackgroundColor (1, 1, 1); //.5, .5, 1
  	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_a_rgb);
  	viewer->addPointCloud<pcl::PointXYZRGB> (cloud_a_rgb, rgb, "sample cloud");
  	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  	
  	
  	pcl::ModelCoefficients cylinder_coeff;
	cylinder_coeff.values.resize (7);    // We need 7 values
  	for(int i=0; i<sphere_locs.size(); i++){
  		//if(i == std::end(to_remove)){ //check if 'i' is one of the cylinders we want to remove
  		if(std::find(to_remove.begin(), to_remove.end(), i) == to_remove.end()) {  //check if 'i' is one of the cylinders we want to remove
	  		std::stringstream ss6;
	  		ss6 << "cylinder" << i;
			cylinder_coeff.values[0] = sphere_locs[i][0];
			cylinder_coeff.values[1] = sphere_locs[i][1];
			cylinder_coeff.values[2] = sphere_locs[i][2];
			cylinder_coeff.values[3] = sphere_locs[i][3] - sphere_locs[i][0];//sphereCloud->points[7].x - sphereCloud->points[1].x;
			cylinder_coeff.values[4] = sphere_locs[i][4] - sphere_locs[i][1];//sphereCloud->points[7].y - sphereCloud->points[1].y;
			cylinder_coeff.values[5] = sphere_locs[i][5] - sphere_locs[i][2];//sphereCloud->points[7].z - sphereCloud->points[1].z;
			cylinder_coeff.values[6] = 0.025;
			//pcl::addCylinder (cylinder_coeff);
			//viewer->addCylinder (cylinder_coeff, ss6.str());
			//viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 1, ss6.str());
			//viewer->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, ss6.str());
			cylinder_coeff.values.clear();
		}
	}
	

	//cout << "x: " << sphereCloud->points[1].x - sphereCloud->points[7].x << ", y: " << sphereCloud->points[1].y - sphereCloud->points[7].y << ", z: " << sphereCloud->points[1].z - sphereCloud->points[7].z << endl;

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	
	
    return(0);

}