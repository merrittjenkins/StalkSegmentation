To run the script:

	./main6
   	This will load 2D stereo pairs, convert them to point clouds, and stitch 5 clouds together. The images are stored in /StalkSegmentation/images2 and writes the point cloud to /StalkSegmentation/stitched_clouds2_complete

	./grow_up14 5
	This loads point clouds stored in /StalkSegmentation/stitched_clouds2_complete and displays the cloud with detected stalks in PCL's viewer. The second argument corresponds to the cloud you want to load.
