# BOLSA
Test's Explanation
\secti
-KNN 5 - Com k = 3, TRAIN = 80, VAL = 1, TEST = 1 
Initial, the idea is test if increasing the number of runs, the model would learn better. Therefore, that was not the case, the 80 runs has the same influence as having the 40, hence, there is no need to have that many. 

-KNN6 Com k = 3, Train = 50, VAL = 1, TEST = 1
Idea: 
-The model will have less train's run. 
-There will be added the following metrics: homogeneity, mutual info, v-measure (homogeneity + completeness) and rand-score. 
-It will be added the image to see where it happens the wrong classification in along each run. (Validation and Test).