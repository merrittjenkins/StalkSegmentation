#include "standard_includes.h"
#include "normaldensity_filter.h"

using namespace std;

void filter(cloud1_up, maxPt, minPt, extension){

	//-------------------Apply filtering based on normals and x-density-----------------------

	float z_distr = maxPt.z-maxPt.z+2*extension;
	float y_distr = maxPt.y-minPt.y+2*extension;

	cout << "number of points: " << cloud1_up->size() << endl;
	cout << "z_distr is: " << z_distr << endl;
	cout << "y_distr is: " << y_distr << endl;

	int no_of_rows = int(y_distr*100)+1; 
	int no_of_cols = int(z_distr*100)+1;
	int initial_value = 0;

	cout << "boundary is: " << no_of_rows << ", " << no_of_cols << endl;

	std::vector<std::vector<int> > matrix;
	std::vector<std::vector<float> > matrix_normals;
	std::map <std::string, std::vector<int> > point_indices; //this is a map that will be filled with a string of the matrix location and the associated point indices

	matrix.resize(no_of_rows, std::vector<int>(no_of_cols, initial_value));
	matrix_normals.resize(no_of_rows, std::vector<float>(no_of_cols, initial_value));

	// Loop through every point in the cloud	
	for(int i=0; i<cloud1_up->size(); i++)
	{
		int scatter_y = -int(((*cloud1_up)[i].y*-100) + (minPt.y - extension)*100); //this is the y-location of the point
		int scatter_z = -int(((*cloud1_up)[i].z*-100) + (minPt.z - extension)*100); //this is the z-location of the point

		matrix[scatter_y][scatter_z] = matrix[scatter_y][scatter_z]+1; //add a count to that cell location
		matrix_normals[scatter_y][scatter_z] = matrix_normals[scatter_y][scatter_z]+std::abs((*cloud1_up)[i].normal_x); //add z-normal to that cell location

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

			if ((matrix[i][j]<40) || (matrix_normals[i][j]>0.5))
			{
				std::stringstream ss;
				ss << i << j;		

				// Iterate through all elements associated with the key and delete them
				for(int k=0; k<point_indices[ss.str()].size(); k++){  //assigning these as 1 is a total hack and needs to be fixed
					(*cloud1_up)[point_indices.find(ss.str())->second[k]].normal_x = nanInsert;
				}
			}
		}
	}

	std::vector<int> indices2;
    pcl::removeNaNNormalsFromPointCloud(*cloud1_up,*cloud1_up, indices2);

    return(*cloud1_up);
}
