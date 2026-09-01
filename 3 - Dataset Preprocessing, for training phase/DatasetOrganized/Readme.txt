
---------------------------------------------------------------------------------------------------------------
- AllData
	- Dataset3Classes
		- images	-> 7229 images
			- test		-> 1420 images
				- BlackRot		-> 508 images
				- Esca			-> 480 images
				- LeafBlight	-> 432 images
			- train		-> 5809 images
				- BlackRot		-> 2168 images
				- Esca			-> 1919 images
				- LeafBlight	-> 1722 images
		- labels	-> 7229 txt
			- test		-> 1420 txt
				- BlackRot		-> 508 txt
				- Esca			-> 480 txt
				- LeafBlight	-> 432 txt
			- train		-> 5809 txt
				- BlackRot		-> 2168 txt
				- Esca			-> 1919 txt
				- LeafBlight	-> 1722 txt
	- DatasetBlackRot
		- images
			- test		-> 508 images
				- BlackRot		-> 508 images
			- train		-> 2168 images
				- BlackRot	-> 2168 images
		- labels
			- test		-> 508 txt
				- BlackRot	-> 508 txt
			- train		-> 2168 txt
				- BlackRot	-> 2168 txt
	- DatasetEsca
		- images
			- test		-> 480 images
				- Esca		-> 480 images
			- train		-> 1919 images
				- Esca		-> 1919 images
		- labels
			- test		-> 480 txt
				- Esca		-> 480 txt
			- train		-> 1919 txt
				- Esca		-> 1919 txt
	- DatasetLeafBlight
		- images
			- test		-> 432 images
				- LeafBlight	-> 432 images
			- train		-> 1722 images
				- LeafBlight	-> 1722 images
		- labels
			- test		-> 432 txt
				- LeafBlight	-> 432 txt
			- train		-> 1722 txt
				- LeafBlight	-> 1722 txt

---------------------------------------------------------------------------------------------------------------
- WithoutOutliers_1_20
	- Dataset3Classes_WithoutOutliers_1_20
		- images
			- test		-> 1416 images
				- BlackRot		-> 504 images
				- Esca			-> 480 images 
				- LeafBlight	-> 432 images
			- train		-> 5788 images
				- BlackRot		-> 2149 images
				- Esca			-> 1917 images
				- LeafBlight	-> 1722 images
		- labels
			- test		-> 1416 txt
				- BlackRot		-> 504 txt
				- Esca			-> 480 txt 
				- LeafBlight	-> 432 txt
			- train		-> 5788 txt
				- BlackRot		-> 2149 txt
				- Esca			-> 1917 txt
				- LeafBlight	-> 1722 txt
	- DatasetBlackRot_WithoutOutliers_1_20
		- images
			- test		-> 504 images
				- BlackRot		-> 504 images
			- train		-> 2149 images
				- BlackRot		-> 2149 images
		- labels
			- test		-> 504 txt
				- BlackRot		-> 504 txt
			- train		-> 2149 txt
				- BlackRot		-> 2149 txt
	- DatasetEsca_WithoutOutliers_1_20
		- images
			- test		-> 480 images 
				- Esca		-> 480 images 
			- train		-> 1917 images
				- Esca		-> 1917 images
		- labels
			- test		-> 480 txt 
				- Esca		-> 480 txt 
			- train		-> 1917 txt
				- Esca		-> 1917 txt
	- DatasetLeafBlight_WithoutOutliers_1_20
		- images
			- test		-> 432 images
				- LeafBlight		-> 432 images
			- train		-> 1722 images
				- LeafBlight		-> 1722 images
		- labels
			- test		-> 432 txt
				- LeafBlight		-> 432 txt
			- train		-> 1722 txt
				- LeafBlight		-> 1722 txt



---------------------------------------------------------------------------------------------------------------
- WithoutOutliers_1_50
	- Dataset3Classes_WithoutOutliers_1_50
		- images
			- test		-> 1418 
				- BlackRot		-> 506 images
				- Esca			-> 480 images
				- LeafBlight	-> 432 images
			- train		-> 5794 
				- BlackRot		-> 2154 images
				- Esca			-> 1918 images
				- LeafBlight	-> 1722 images
		- labels
			- test
				- BlackRot		-> 506 txt
				- Esca			-> 480 txt
				- LeafBlight	-> 432 txt
			- train
				- BlackRot		-> 2154 txt
				- Esca			-> 1918 txt
				- LeafBlight	-> 1722 txt
	- DatasetBlackRot_WithoutOutliers_1_50
		- images
			- test		-> 506 images
				- BlackRot	-> 506 images
			- train		-> 2154 images
				- BlackRot		-> 2154 images
		- labels
			- test		-> 506 txt
				- BlackRot		-> 2154	txt	 
			- train		-> 2154 txt
				- BlackRot		-> 2154 txt
	- DatasetEsca_WithoutOutliers_1_50
		- images
			- test		-> 480 images
				- Esca		-> 480 images
			- train		-> 1918 images
				- Esca		-> 1918 images
		- labels		
			- test		-> 480 txt
				- Esca		-> 480 txt
			- train		-> 1918 txt
				- Esca		-> 1918	txt
	- DatasetLeafBlight_WithoutOutliers_1_50
		- images
			- test		-> 432 images
				- LeafBlight	-> 432 images
			- train		-> 1722 images
				- LeafBlight	-> 1722 images
		- labels
			- test		-> 432 txt
				- LeafBlight	-> 432 txt
			- train		-> 1722 txt
				- LeafBlight	-> 1722 txt





---------------------------------------------------------------------------------------------------------------
BlackRot
	Train
		Annot - Number of With Outliers: 17685
		Annot - Number of Without Outliers (1.2): 16731
		Annot - Number of Without Outliers (1.5): 16990

	Test
		Annot - Number of With Outliers: 3918
		Annot - Number of Without Outliers (1.2): 3424
		Annot - Number of Without Outliers (1.5): 3567

Esca
	Train
		Annot - Number of With Outliers: 24491
		Annot - Number of Without Outliers (1.2): 22266
		Annot - Number of Without Outliers (1.5): 22723

	Test
		Annot - Number of With Outliers: 5098
		Annot - Number of Without Outliers (1.2): 4662
		Annot - Number of Without Outliers (1.5): 4753

Leaf Blight
	Train
		Annot - Number of With Outliers: 28351
		Annot - Number of Without Outliers (1.2): 25841
		Annot - Number of Without Outliers (1.5): 26308

	Test
		Annot - Number of With Outliers: 5173
		Annot - Number of Without Outliers (1.2): 4652
		Annot - Number of Without Outliers (1.5): 4736

