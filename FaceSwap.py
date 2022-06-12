import cv2 as cv
import numpy as np
import cmake
import dlib
def index_from_array(arr):
    for n in arr[0]:
        index=n
        break;
    return index
triangle_index_points_list=[]      
frontal_face_detector=dlib.get_frontal_face_detector()
frontal_shape_predictor=dlib.shape_predictor("D:\DeepFake_Practice\FaceSwap\DataSet\shape_predictor_68_face_landmarks.dat")
imu_bgr=cv.imread("D:\DeepFake_Practice\FaceSwap\image\imu.jpg")
imu=cv.cvtColor(imu_bgr,cv.COLOR_BGR2GRAY)
#cv.imshow("Imran Khan",imu)
putin_bgr=cv.imread("D:\DeepFake_Practice\FaceSwap\image\putin.jpg")
putin=cv.cvtColor(putin_bgr,cv.COLOR_BGR2GRAY)
# cv.imshow("Validemer Putin",putin)
source_image_canvas=np.zeros_like(imu)
height,width,no_of_channels=putin_bgr.shape
destination_image_canvas=np.zeros((height,width,no_of_channels),np.uint8)
#FOR SOURCE
source_faces=frontal_face_detector(imu)
for source_face in source_faces:
    source_face_landmarks=frontal_shape_predictor(imu,source_face)
    source_face_landmark_points=[]
    for landmark_no in range(0,68):
        x_point=source_face_landmarks.part(landmark_no).x
        y_point=source_face_landmarks.part(landmark_no).y
        source_face_landmark_points.append((x_point,y_point))
        #cv.circle(imu_bgr,(x_point,y_point),2,(255,0,0),-1)
        #cv.imshow("landmarked",imu_bgr)
    source_face_landmark_points_array=np.array(source_face_landmark_points,np.int32)
    source_face_convexhull=cv.convexHull(source_face_landmark_points_array)
    # cv.polylines(imu_bgr, [source_face_convexhull], True, (255,0,0),3)
    # cv.imshow("Convex Hull",imu_bgr)
    cv.fillConvexPoly(source_image_canvas,source_face_convexhull,255)
    #cv.imshow("Canvass with convex hull",source_image_canvas)
    source_face_image=cv.bitwise_and(imu_bgr, imu_bgr,mask=source_image_canvas)
    #cv.imshow("place mask over source image",source_face_image)
#FOR DESTINATION   
destination_faces=frontal_face_detector(putin)
for destination_face in destination_faces:
    destination_face_landmarks=frontal_shape_predictor(putin,destination_face)
    destination_face_landmark_points=[]
    for landmark_no in range(0,68):
        x_point=destination_face_landmarks.part(landmark_no).x
        y_point=destination_face_landmarks.part(landmark_no).y
        destination_face_landmark_points.append((x_point,y_point))
        #cv.circle(imu_bgr,(x_point,y_point),2,(255,0,0),-1)
        #cv.imshow("landmarked",imu_bgr)
    destination_face_landmark_points_array=np.array(destination_face_landmark_points,np.int32)
    destination_face_convexhull=cv.convexHull(destination_face_landmark_points_array)
    # cv.polylines(putin_bgr, [destination_face_convexhull], True, (255,0,0),3)
    # cv.imshow("Convex Hull1",putin_bgr)
bounding_rectangle=cv.boundingRect(source_face_convexhull)
subdivisions=cv.Subdiv2D(bounding_rectangle)
subdivisions.insert(source_face_landmark_points)
triangles_vector=subdivisions.getTriangleList()
triangles_array=np.array(triangles_vector,dtype=np.int32)
for triangle in triangles_array:
    index_point1=(triangle[0],triangle[1])
    index_point2=(triangle[2],triangle[3])
    index_point3=(triangle[4],triangle[5])
    line_color=(255,0,0)
    cv.line(source_face_image,index_point1,index_point2,line_color,1)
    cv.line(source_face_image,index_point2,index_point3,line_color,1)
    cv.line(source_face_image,index_point3,index_point1,line_color,1)
    #cv.imshow("gg",source_face_image)
    index_point1=np.where((index_point1==source_face_landmark_points_array).all(axis=1))
    index_point2=np.where((index_point2==source_face_landmark_points_array).all(axis=1))
    index_point3=np.where((index_point3==source_face_landmark_points_array).all(axis=1))
    index_point1=index_from_array(index_point1)
    index_point2=index_from_array(index_point2)
    index_point3=index_from_array(index_point3)
    triangle=[index_point1,index_point2,index_point3]
    triangle_index_points_list.append(triangle)

for i,triangle_index_points in enumerate(triangle_index_points_list):
    source_triangle_point1=source_face_landmark_points[triangle_index_points[0]]
    source_triangle_point2=source_face_landmark_points[triangle_index_points[1]]
    source_triangle_point3=source_face_landmark_points[triangle_index_points[2]]
    source_triangle=np.array([source_triangle_point1,source_triangle_point2,source_triangle_point3],np.int32)
    source_rectangle=cv.boundingRect(source_triangle)
    (x,y,w,h)=source_rectangle
    cropped_source_rectangle=imu_bgr[y:y+h,x:x+w]
    if i==10:
        cv.line(imu_bgr,source_triangle_point1,source_triangle_point2,(255,255,255))
        cv.line(imu_bgr,source_triangle_point2,source_triangle_point3,(255,255,255))
        cv.line(imu_bgr,source_triangle_point3,source_triangle_point1,(255,255,255))
        #cv.imshow("10th",imu_bgr)
        cv.rectangle(imu_bgr, (x,y), (x+w,y+h), (0,0,255))
        #cv.imshow("Rect",imu_bgr)
        #cv.imshow("Croped",cropped_source_rectangle)
    destination_triangle_point1=destination_face_landmark_points[triangle_index_points[0]]
    destination_triangle_point2=destination_face_landmark_points[triangle_index_points[1]]
    destination_triangle_point3=destination_face_landmark_points[triangle_index_points[2]]
    destination_triangle=np.array([destination_triangle_point1,destination_triangle_point2,destination_triangle_point3],np.int32)
    destination_rectangle=cv.boundingRect(destination_triangle)
    (x,y,w,h)=destination_rectangle
    cropped_destination_rectangle=imu_bgr[h,w]
    cropped_destination_rectangle_mask=np.zeros((h,w),np.uint8)
    destination_triangle_points=np.array([[destination_triangle_point1[0]-x,destination_triangle_point1[1]-x],
                                          [destination_triangle_point2[0]-x,destination_triangle_point2[1]-x],
                                          [destination_triangle_point3[0]-x,destination_triangle_point3[1]-x]],np.int32)
    cv.fillConvexPoly(cropped_destination_rectangle_mask,destination_triangle_points,255)
    if i==10:
        cv.line(putin_bgr,destination_triangle_point1,destination_triangle_point2,(255,255,255))
        cv.line(putin_bgr,destination_triangle_point2,destination_triangle_point3,(255,255,255))
        cv.line(putin_bgr,destination_triangle_point3,destination_triangle_point1,(255,255,255))
        cv.imshow("10th1",putin_bgr)
        cv.rectangle(putin_bgr, (x,y), (x+w,y+h), (0,0,255))
        cv.imshow("Rect1",putin_bgr)
        cv.imshow("Croped1",cropped_destination_rectangle_mask)