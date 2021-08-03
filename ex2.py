import numpy
from sklearn import datasets 
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt 
from sklearn import datasets, svm, metrics
from sklearn import linear_model
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D 
import array as arr
# ------------------------------------------------------------
# The function corrects the digits and displays all the unprinted 
# digits properly for proper printing of the digits
# The function checks which digits are not printed correctly
# And then passes the patch to the print function show
def get_list_wrongs(my_data):
    wrongs = list()
    for index in my_data:
        if index[1] != index[2]:
            wrongs.append(index)
    show(wrongs,0,None)
# ------------------------------------------------------------
# the function sum the all row on the matrix and  
# return the all sum of the rows on the matrix
def sum_matrix(matrix):
     index = 0
     my_list = [0 for i in matrix]    
     for i in matrix:
          my_list[index] = [numpy.sum(i)]
          index += 1
     return my_list
# ------------------------------------------------------------
# the function return the center of the matrix values
# the function return The sum of the central area of the matrix
def sum_of_center_mat(matrix):
     index = 0
     my_list = [0 for i in matrix]   
     mid = len(matrix[0]) // 2 - 1
     for mat in matrix:
          for col in range(mid, mid + 2):
               for row in range(mid, mid + 2):
                    my_list[index] += mat[col][row]
          index += 1
     return my_list
# ------------------------------------------------------------
# the function getting the matrix and return the center of values
def get_center(matrix,mid):
    center = 0
    for col in range(mid, mid + 2):
        for row in range(mid, mid + 2):
            center += matrix[col][row]
    return center
# ------------------------------------------------------------
# The function returns the sum of the values   
# of the same column in the matrix
# The function returns the sum of all the values of the same 
# column in the matrix of each column and column found in the matrix
def sum_matrix_cols(matrix):
     mid = len(matrix[0]) // 2 - 1
     index = 0
     my_list = [0 for i in matrix]        
     for mat in matrix:
          sum = 0
          for col in range(mid, mid + 2):
               for row in range(len(matrix[0])):
                    my_list[index] += mat[row][col]                 
          index += 1     
     return my_list
# ------------------------------------------------------------
# The function returns the perimeter of the matrix
def get_perimeter(matrix):
     perimeter = 0
     for i in range(1, len(matrix[0])):
        perimeter += matrix[i][0]
     for i in range(1, len(matrix[0])):
        perimeter += matrix[i][-1]
     return perimeter
# ------------------------------------------------------------
# The function compares the extent of the matrix 
# with the sum of the matrix values
def matrix_center_vs_perimeter(matrix):
     mid = len(matrix[0]) // 2 - 1
     index = 0     
     my_list = [0 for i in matrix]  
     for i in matrix:
          perimeter = numpy.sum(i[0])
          perimeter += numpy.sum(i[-1])
          perimeter += get_perimeter(i)
          my_list[index] = abs(get_center(i,mid) - perimeter)
          index += 1
     return my_list
# -------------------------------------------------------------
# the function sum the rows and the cols on the 
# matrix the function return cols - rows
def sum_up_down(matrix,size,up,down):
     index = 0
     for row in matrix:
            if index < size:
                up += numpy.sum(row)
            else:
                down += numpy.sum(row)
            index += 1
     return (up - down)
# -------------------------------------------------------------
# the function sum the rows and the cols on the 
# matrix the function return the abs(cols - rows) 
def sum_of_up_vs_down_matrix(matrix):
     index = 0
     up = 0 
     down = 0
     my_list = [0 for i in matrix]  
     for i in matrix:
          my_list[index] = sum_up_down(i,4,up,down)
          index += 1
     return my_list
# -------------------------------------------------------------
# The function returns the difference between quadrants   
# of the matrix after checking for each row and each    
# column that are values in the requested matrix The    
# values then go into each line that matches it
def num_of_difference(matrix,flag):
     index, qrtr1, qrtr2, qrtr3, qrtr4 = 0, 0, 0, 0, 0
     mid = len(matrix[0]) // 2
     my_list = [0 for i in matrix]
     if flag is False:
          for i in matrix:
               qrtr1, qrtr2, qrtr3, qrtr4 = 0, 0, 0, 0
               for row in range(0, mid):
                    for col in range(0, mid):
                         qrtr1 += i[row][col]
               for row in range(0, mid):
                    for col in range(mid, len(matrix[0])):
                         qrtr2 += i[row][col]
               for row in range(mid, len(matrix[0])):
                    for col in range(0, mid):
                         qrtr3 += i[row][col]
               for row in range(mid, len(matrix[0])):
                    for col in range(mid, len(matrix[0])):
                         qrtr2 += i[row][col]
               my_list[index] = abs((qrtr1 + qrtr3) -  (qrtr2 + qrtr4))              
               index += 1
          i = 10
     else:
          for i in matrix:
               for j in i:
                    my_list[index] += j[1] + j[-2]
               index += 1
     return my_list
# -------------------------------------------------------------
# The program display function prints all the values received 
# according to the parameters sent ie for each question in the program
def show(features, names, flag):
     if flag is None:
          plt.figure(figsize = (8,6), facecolor = "tab:gray")
          for index,(img, labl, pred) in enumerate(features):
               plt.subplot(3, 10, index+1)
               plt.axis('off')
               plt.imshow(img, cmap = plt.cm.gray_r, interpolation = 'nearest')
               plt.title('{} {}'.format(labl, pred))
          plt.suptitle('Test.mis-classification: expected-predicted')
          plt.show()
     else:
          for f1 in range(0, len(features) - 2):
               for f2 in range(f1 + 1, len(features) - 1):
                    if (flag is True):
                         for f3 in range(f2 + 1, len(features)):
                              fig = plt.figure(figsize = (9,7))
                              ax = fig.add_subplot(111, projection='3d')
                              ax.scatter(features[f1], features[f2], features[f3], c=digits.target[indices_0_1])
                              ax.set_xlabel(names[f1])
                              ax.set_ylabel(names[f2])
                              ax.set_zlabel(names[f3])
                              plt.show()
                    else:
                         plt.figure(figsize = (9,7))
                         plt.scatter(features[f1], features[f2], c = digits.target[indices_0_1])
                         plt.xlabel(names[f1])
                         plt.ylabel(names[f2])
                         plt.show()

# =========================== Main ============================
# ======================= Question 20 =========================
# The function corrects the digits and displays all the unprinted 
# digits properly for proper printing of the digits
# The function checks which digits are not printed correctly
# And then passes the patch to the print function show
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))
my_class = svm.SVC(gamma = 0.001)
my_class.fit(data[:len(digits.images) // 2] , digits.target[:len(digits.images) // 2])
predicted = my_class.predict(data[len(digits.images) // 2:])
my_images = list(zip(digits.images[len(digits.images) // 2:], digits.target[len(digits.images) // 2:], predicted))
get_list_wrongs(my_images)
# ======================= Question 21 =========================
# 
indices_0_1 = numpy.where(numpy.logical_and(digits.target >= 0, digits.target <= 1))
my_image = digits.images[indices_0_1]
sum_mat = sum_matrix(my_image)
m_1 = sum_matrix_cols(my_image)  
m_2 = sum_of_center_mat(my_image)
m_3 = matrix_center_vs_perimeter(my_image)
m_4 = sum_of_up_vs_down_matrix(my_image)
m_5 = num_of_difference(my_image,False)
m_6 = num_of_difference(my_image,True)
list_of_feat = [m_1,m_2,m_3,m_4,m_5,m_6]
list_of_names = ['var x', 'var y', 'digit', 'var x', 'var y', 'digit']
show(list_of_feat, list_of_names,False)
show(list_of_feat, list_of_names,True)
# ====================== Question 21-F ========================
# creating the X (feature) matrix 
X = numpy.column_stack((m_4,m_5,m_2)) 
# scaling the values for better classification performance 
X_scaled = preprocessing.scale(X) 
# the predicted outputs 
Y = digits.target[indices_0_1]
# Training Logistic regression 
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(X_scaled, Y) 
# show how good is the classifier on the training data 
expected = Y  
predicted = logistic_classifier.predict(X_scaled)
print("Logistic regression using [featureA, featureB] features:\n%s\n" % 
      (metrics.classification_report(expected,predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)) 
# estimate the generalization performance using cross validation
predicted2 = cross_val_predict(logistic_classifier, X_scaled, Y, cv=10) 
print("Logistic regression using [featureA, featureB] features cross validation:\n%s\n" % 
      (metrics.classification_report(expected,predicted2))) 
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2)) 
# ==============================================================