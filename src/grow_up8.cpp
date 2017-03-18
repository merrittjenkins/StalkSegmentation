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

	ofstream outFile;
	outFile.open("users.dat");

	float nanInsert = std::numeric_limits<float>::quiet_NaN();

	int temp=0;
	int I;
	int J;
	int counter = 0;
	//float Z;
	//float Y; 
	cout << "No of rows: " << no_of_rows << endl;
	cout << "No of cols: " << no_of_cols << endl;
	// THIS TAKES THE AVERAGE NORMAL VECTOR FOR EACH VOXEL AND THEN DECIDES TO KEEP OR TOSS THE ELEMENTS OF THE VOXEL
	for(int i=0; i<no_of_rows; i++)
	{
		for(int j=0; j<no_of_cols; j++) 
		{
			if ((matrix_normals[i][j]) > 0)
			{
				matrix_normals[i][j] = (matrix_normals[i][j])/(matrix[i][j]);
			}


			if ((matrix[i][j]>100) && (matrix_normals[i][j]<0.5)) {
				outFile << i << " " << j << " " << matrix[i][j] << "\n";
				/*
				if (matrix[i][j] > temp) {
					temp = matrix[i][j];
					I = i;
					J = j; 
				}
				*/
				//cout << "HERE: " << endl;
				// Add point to hotspots vector
				hotspots[counter][0] = minPt.z + float(j)/100;
				hotspots[counter][1] = minPt.y + float(i)/100;

				cout << "y location: " << hotspots[counter][1] << endl;

				// Decimate surroundings
				
				for(int k=-15; k<16; k++){
					for(int l=-8; l<9; l++){
						if((k+i<no_of_rows)&&(k+i>=0)&&(l+j<no_of_cols)&&(l+j>=0)) {
							//cout << "row to decimate: " << k+i << ", col to decimate: " << l+j << endl;
							matrix[k+i][l+j] = 0;
						}
					}
				}
				

				counter++;

			}
			else{
				outFile << i << " " << j << " " << 0 << "\n";				
			}



			/*

			if ((matrix[i][j]<15) || (matrix_normals[i][j]>0.5))
			{
				std::stringstream ss;
				ss << i << j;		

				// Iterate through all elements associated with the key and delete them
				for(int k=0; k<point_indices[ss.str()].size(); k++){  //assigning these as 1 is a total hack and needs to be fixed
					(*cloud)[point_indices.find(ss.str())->second[k]].normal_x = nanInsert;
				}
			}
			*/
		}
	}

	/*
	cout << "TEMP IS: " << temp << endl;
	// Convert rows and columns of the heat map back to point locations

	cout << "maxPt.z is: " << maxPt.z << endl;
	cout << "z offset: " << float(J)/100 << endl;
	cout << "J: " << J << endl; 
	float Z = minPt.z + float(J)/100;
	float Y = float(I)/100+minPt.y;
	hotspots[0][0] = Z;
	hotspots[0][1] = Y;
	*/

	//std::vector<int> indices2;
    //pcl::removeNaNNormalsFromPointCloud(*cloud,*cloud, indices2);

    //cout << "ALSO HERE" << endl;

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
    reader.read<pcl::PointXYZ> ("stitched_clouds2/plants_tform_106.ply", *cloud_a);
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

    for(int n=0; n<1; n++){
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
	    //writer.write ("slice.ply", *cloud_slice, false);

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
	
		/*
		// write just the filtered slice (no region growing yet)
	    writer.write ("slice_filtered.ply", *cloud_slice, false);

	    // Perform region growing on the sliced point cloud
		multiCloud = regionGrow_ground(cloud_slice); //outputs a vector of clouds

		cout << "Number of clouds: " << multiCloud.size() << endl;
		*/

		//UNCOMMENT THIS TO WRITE THE GROUND-SLICE CLOUD CLUSTERS TO DISK
		/*
		for(int j=0; j<multiCloud.size(); j++){
			std::cout << "PointCloud representing the Cluster: " << (multiCloud[j])->points.size () << " data points." << std::endl;
			std::stringstream ss;
			ss << "cloud_cluster_" << n << "_" << j << ".ply";
			writer.write<pcl::PointXYZ> (ss.str (), *(multiCloud[j]), false);
		}
		duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    	cout<<"TIME, FILTERING: " << duration << endl;
    	*/
	}


	/*
	//----------------------------------THIS IS THE START OF GROWING UPWARDS--------------------------------

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_a_normals (new pcl::PointCloud<pcl::PointNormal>);
	std::map< int, std::map<int, std::vector<float> > > stalk_centroids2; // Create a map that will hold the stalk centroids

	int no_of_slices = 6;
	float extension = 0.035; //extended boundary of each upward growth
	int no_sections_threshold = 2; //there must be more than two recognized sections to be considered a stalk
	int min_stalk_points = 100;
	float slice_height = -0.20;


	for(int k=0; k<multiCloud.size(); k++) {
	    
	    // Load one of the presumed stalks
	    cloud1 = multiCloud[k];
	    ////cout << "Size of imported cloud: " << cloud1->size() << endl;

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
			condrem.setInputCloud(cloud_normals);
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
	*/

	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"Duration: "<< duration <<'\n';

	// Display the points
	pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_a_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read<pcl::PointXYZRGB> ("stitched_clouds2/plants_tform_106.ply", *cloud_a_rgb);
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