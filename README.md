# Template matching in depth images

Template matching is a very practical technique for finding all the occurrences of a known target in a given query image. Deep-learning techniques are not very flexible, requiring re-training for new templates. Traditional computer vision techniques (e.g. SIFT + RANSAC) are more practical but require an underlying model (homography) describing the transformation between the template and its occurrences in the query image. When there are several occurrences of the same template in the scene, template matching becomes a robust multi-model fitting problem, which requires to disambiguate among multiple spurious matches.

## Goal
The goal of this project is to expand a template detection pipeline based on images only, to RGB-D data. Point clouds should be leveraged to constraints the homography search in regions belonging to the same planar surface in the scene. Another research direction would be to adaptively identify which template to search first in the target image, which is a very relevant problem when the number of templates is very large (e.g. a few hundreds) by exploiting the 3D data of the scene.
