import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process
from tkinter import *
import pickle
import numpy as np
import cv2
import os
import time
import cv2
import numpy as np
cx = 0
cy = 0
    
def heat_map(img,points,gauss,k=21,height=700,width=700,tim=5):
    if len(points)>tim:
        print("lengthhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",len(points))
        points=points[len(points)-1-tim:]
    for point in points:
        for p in point:
            x=p[0]
            y=p[1]
            xi=int((x-10))
            xj=int((x+10))+1
            yi=int((y-10))
            yj=int((y+10))+1
            # print("gauss looooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo",xi,xj,yi,yj,np.array(img[xi:xj,yi:yj,:]).shape)
            img[xi:xj,yi:yj,:]=img[xi:xj,yi:yj,:]*gauss
            # print("updtedddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",img.shape)
            # b=img1[xi:xj,yi:yj,:]
            # c=gauss+b
            # img[xi:xj,yi:yj,:]=img[xi:xj,yi:yj,:]+c
        # img=img/(np.max(img,axis=(0,1)))+0.0001
    return img
def gaussian_kernel(length,mean_):
    gauss_kernel=cv2.getGaussianKernel(length,mean_)
    
    gauss_kernel=gauss_kernel*gauss_kernel.T
    gauss_kernel=gauss_kernel/gauss_kernel[(np.int(length/2)),np.int(length/2)]
    gauss_kernel=cv2.cvtColor(cv2.applyColorMap((gauss_kernel*255).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    print("max gaussian",np.max(gauss_kernel))
    return gauss_kernel
    

def alert(title,msg):
    root = Tk()
    root.title("خبردار")
    w = 400
    h = 400
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x  = (ws - w)/2
    y = (hs - h) /2
    root.geometry('%dx%d+%d+%d' %(w,h,x,y))
    l = Label(root,text=msg,width=100,height=10)
    l.pack()
    b = Button(root,text="ok",command=root.destroy,width=10)
    b.pack()
    mainloop()
def check_detection(obj,rect):
    
    detected = False
    x = obj[0]
    y = obj[1]
    rxl = rect[0]
    ryl = rect[1]
    rxh = rect[2]
    ryh = rect[3]
    if x >= rxl and x <= rxh and y >=ryl and y <= ryh:
        detected = True
    return detected


def draw_rectangle(event,x,y,flags,param):
    color = (0,0,0)
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(merged_view,(x,y),(x+50,y+50),color,2)
        clicked_points.append([x,y])
        

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        if param[0] == "top_view1":
            top_view1_points.append([x, y])
            print('top_view1', len(top_view1_points))

            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            cv2.drawMarker(top_view1_copy, coordinates, color, markerSize=ms)
            cv2.imshow("top_view1", top_view1_copy)
            print("top1:", x, ' ', y)
        if param[0] == "top_view2":
            top_view2_points.append([x, y])
            print('top_view2', len(top_view2_points))

            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            cv2.drawMarker(top_view2_copy, coordinates, color, markerSize=ms)
            cv2.imshow("top_view2", top_view2_copy)
            print("top2:", x, ' ', y)
        if param[0] == "top_view3":
            top_view3_points.append([x, y])
            print('top_view3', len(top_view3_points))
            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            cv2.drawMarker(top_view3_copy, coordinates, color, markerSize=ms)
            cv2.imshow("top_view3", top_view3_copy)
            print("top3:", x, ' ', y)
        
        if param[0] == "view1":
            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            view1_points.append([x, y])
            print('view1', len(view1_points))

            cv2.drawMarker(view1_copy, coordinates, color, markerSize=ms)
            print("view1", x, ' ', y)
            cv2.imshow("view1", view1_copy)
        if param[0] == "view2":
            view2_points.append([x, y])
            print('view2', len(view2_points))

            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            cv2.drawMarker(view2_copy, coordinates, color, markerSize=ms)
            cv2.imshow("view2", view2_copy)
            print("view2:", x, ' ', y)
        if param[0] == "view3":
            view3_points.append([x, y])
            print('view3', len(view3_points))

            coordinates = (x, y)
            color = (255, 0, 0)
            ms = 5
            cv2.drawMarker(view3_copy, coordinates, color, markerSize=ms)
            cv2.imshow("view3", view3_copy)
            print("view3:", x, ' ', y)

def project_midpoint_into_top_view(point,H):
    new_point = np.dot(H,np.array(point))
    new_point=new_point/new_point[2]
    
    print("newew",new_point)
    return new_point


def save_boxes(boxes1,boxes2,boxes3):
    with open("boxes1.data","wb") as f:
        pickle.dump(boxes1,f)
    with open("boxes2.data","wb") as f:
        pickle.dump(boxes2,f)
    with open("boxes3.data","wb") as f:
        pickle.dump(boxes3,f)
    print("boxes saved!")
def load_boxes():
    with open("boxes1.data","rb") as f:
        boxes1 = pickle.load(f)
    with open("boxes2.data","rb") as f:
        boxes2 = pickle.load(f)
    with open("boxes3.data","rb") as f:
        boxes3 = pickle.load(f)
    print("boxes loaded!")
    return boxes1,boxes2,boxes3
def merge_views(v1,v2,v3):

    v1_new = np.where(v1==0,v1,255)
    v2_new = np.where(v2==0,v2,255)
    v3_new = np.where(v3==0,v3,255)
    mask12 = np.ones(v1.shape, dtype=bool)
    mask13 = np.ones(v1.shape, dtype=bool)
    mask23 = np.ones(v1.shape, dtype=bool)
    v1_255 = np.where(v1_new == 255)
    v2_255 = np.where(v2_new == 255)
    v3_255 = np.where(v3_new == 255)
    v1_v2_common_indices = np.where(v1_255 == v2_255)
    v2_v3_common_indices = np.where(v2_255 == v3_255)
    v1_v3_common_indices = np.where(v1_255 == v3_255)
    mask12[v1_v2_common_indices] = True
    mask23[v2_v3_common_indices] = True
    mask13[v1_v3_common_indices] = True
    v1_zero = np.where(v1_new==0)
    v2_zero = np.where(v2_new ==0)
    v3_zero = np.where(v3_new ==0)
    print("v1_new",v1_new.shape,v2_new.shape)
    print("v1",v1[v1_zero].shape,"v2",v2[v2_zero].shape, "v12_common",v1[v1_v2_common_indices].shape,v2[v1_v2_common_indices].shape)
    mean_v1_v2 = (v1[mask12]/255 + v2[mask12]/255)/2
    mean_v2_v3 = (v3[mask23]/255 + v2[mask23]/255)/2
    mean_v1_v3 = (v1[mask13]/255 + v3[mask13]/255)/2  
    merged_view= np.zeros(v1_new.shape)
    merged_view[mask12] = mean_v1_v2
    merged_view[mask13] = mean_v1_v3
    merged_view[mask23] = mean_v2_v3
    merged_view[~mask12] = v1[~mask12]/255 + v2[~mask12]/255
    merged_view[~mask13] = v1[~mask13]/255 + v3[~mask13]/255
    merged_view[~mask23] = v2[~mask23]/255 + v3[~mask23]/255
    return merged_view
    
###################
#Getting corresponding points
###################

#1. Variables initialization
length=21
mean_=8
H=700
W=700
gauss_kernel=gaussian_kernel(length,mean_)
view1_points = []
view2_points = []
view3_points = []
clicked_points = []
top_view1_points = []
top_view2_points = []
top_view3_points = []
#2. Getting Files Paths
top_view_path = os.getcwd() + "/views/top.jpeg"
view1_path = os.getcwd() + "/views/middle.jpg"
view2_path = os.getcwd() + "/views/front.jpg"
view3_path = os.getcwd() + "/views/right.jpg"
print(view1_path)
#3.Reading Files
top_view = cv2.imread(top_view_path)
view1 = cv2.imread(view1_path)
view2 = cv2.imread(view2_path)
view3 = cv2.imread(view3_path)
top_view1_copy = cv2.imread(top_view_path)
top_view2_copy = cv2.imread(top_view_path)
top_view3_copy = cv2.imread(top_view_path)
view1_copy = cv2.imread(view1_path)
view2_copy = cv2.imread(view2_path)
view3_copy = cv2.imread(view3_path)
boxes1 = []
boxes2 = []
boxes3 = []



#########
#B) Code for finding Corresponding points
#########



saved = True

#1. If points are already saved, don't recompute, otherwise compute them
if not saved:
    cv2.namedWindow('top_view1',cv2.WINDOW_AUTOSIZE)
    cv2.imshow("top_view1",top_view)
    cv2.namedWindow('view1',cv2.WINDOW_AUTOSIZE)
    cv2.imshow("view1",view1)
    param_top1 = ["top_view1"]
    cv2.setMouseCallback("top_view1",click_event,param_top1)
    param_view1 = ["view1"]
    cv2.setMouseCallback("view1",click_event,param_view1)
    q=-21
    q = cv2.waitKey(0)
    print(q)
    view1_done = False
    print(q)
    if q > 0:
        view1_done = True
        cv2.destroyAllWindows()
    view2_done =False 
    if view1_done:
        cv2.namedWindow('top_view2',cv2.WINDOW_AUTOSIZE)
        cv2.imshow("top_view2",top_view)
        cv2.namedWindow('view2',cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view2",view2)
        param_top2 = ["top_view2"]
        param_view2 = ["view2"]
        cv2.setMouseCallback("view2",click_event,param_view2)
        cv2.setMouseCallback("top_view2",click_event,param_top2)
        q=-12
        q = cv2.waitKey(0)
        if q>0:
            view2_done = True
            cv2.destroyAllWindows()
    if view2_done:
        cv2.namedWindow('top_view3',cv2.WINDOW_AUTOSIZE)
        cv2.imshow("top_view3",top_view)
        cv2.namedWindow('view3',cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view3",view3)
        param_top3 = ["top_view3"]
        cv2.setMouseCallback("top_view3",click_event,param_top3)
        param_view3 = ["view3"]
        cv2.setMouseCallback("view3",click_event,param_view3)
        q=-12
        q = cv2.waitKey(0)
        if q > 0:
            cv2.destroyAllWindows()

    top_view1_points_path = os.getcwd() + "/points/top_view1_pts.data"
    top_view2_points_path = os.getcwd() + "/points/top_view2_pts.data"
    top_view3_points_path = os.getcwd() + "/points/top_view3_pts.data"
    view1_points_path = os.getcwd() + "/points/middle.data"
    view2_points_path = os.getcwd() + "/points/front.data"
    view3_points_path = os.getcwd() + "/points/right.data"
# Saving points after dumping them
    with open(top_view1_points_path,'wb' ,) as f:
        pickle.dump(top_view1_points,f)
    with open(top_view2_points_path,'wb' ,) as f:
        pickle.dump(top_view2_points,f)
    with open(top_view3_points_path,'wb' ,) as f:
        pickle.dump(top_view3_points,f)
    with open(view1_points_path,'wb' ,) as f:
        pickle.dump(view1_points,f)
    with open(view2_points_path,'wb' ,) as f:
        pickle.dump(view2_points,f)
    with open(view3_points_path,'wb' ,) as f:
        pickle.dump(view3_points,f)
#C) Reading saved points
top_view1_points_path = os.getcwd() + "/points/top_view1_pts.data"
top_view2_points_path = os.getcwd() + "/points/top_view2_pts.data"
top_view3_points_path = os.getcwd() + "/points/top_view3_pts.data"
view1_points_path = os.getcwd() + "/points/middle.data"
view2_points_path = os.getcwd() + "/points/front.data"
view3_points_path = os.getcwd() + "/points/right.data"
with open(top_view1_points_path,'rb' ,) as f:
    top_view1 = pickle.load(f)
with open(top_view2_points_path,'rb' ,) as f:
    top_view2 = pickle.load(f)
with open(top_view3_points_path,'rb' ,) as f:
    top_view3 = pickle.load(f)
with open(view1_points_path,'rb' ,) as f:
    v1 = pickle.load(f)
with open(view2_points_path,'rb' ,) as f:
    v2 = pickle.load(f)
with open(view3_points_path,'rb' ,) as f:
    v3 = pickle.load(f)
#####
#Setup for YOLO Neural network
#####
yolo_weights_path = os.path.join(os.getcwd(),"yolov3.weights")
yolo_cfg_path = os.path.join(os.getcwd(),"yolov3.cfg")
net1 = cv2.dnn.readNet(yolo_weights_path,yolo_cfg_path)
net2 = cv2.dnn.readNet(yolo_weights_path,yolo_cfg_path)
net3 = cv2.dnn.readNet(yolo_weights_path,yolo_cfg_path)
layer1_names = net1.getLayerNames()
layer2_names = net2.getLayerNames()
layer3_names = net3.getLayerNames()
outputlayers1 = [layer1_names[i[0] - 1] for i in net1.getUnconnectedOutLayers()]
outputlayers2 = [layer2_names[i[0] - 1] for i in net2.getUnconnectedOutLayers()]
outputlayers3 = [layer3_names[i[0] - 1] for i in net3.getUnconnectedOutLayers()]
classes = []

#####
#Code for implement object detection
#####

#1.Reading class names
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

#2.Colors for boxes
colors= np.random.uniform(0,255,size=(len(classes),3))
font = cv2.FONT_HERSHEY_SIMPLEX

#3. Computing homography matrices
H1 = cv2.findHomography(np.array(v1), np.array(top_view1))[0]
H2 = cv2.findHomography(np.array(v2), np.array(top_view2))[0]
H3 = cv2.findHomography(np.array(v3), np.array(top_view3))[0]

#4. Getting viewes dimensions
h1,w1,d1 = view1.shape
h2,w2,d3 = view2.shape
h3,w3,d3 = view3.shape



#7. Getting URLs and frames for camera recodrings
boxes1=[]
boxes2=[]
boxes3=[]
url1 = os.getcwd() + "/recordings/middle.avi"
url2 = os.getcwd() + "/recordings/front.avi"
url3 = os.getcwd() + "/recordings/right.avi"
print("url is as following: ",url1)
cap1 = cv2.VideoCapture(url1)
cap2 = cv2.VideoCapture(url2)
cap3 = cv2.VideoCapture(url3)

#8. Setup for displaying video frames
frame1_width = int (cap1.get(3)) 
frame1_height = int (cap1.get(4))
frame2_width = int (cap2.get(3)) 
frame2_height = int (cap2.get(4))
frame3_width = int (cap3.get(3)) 
frame3_height = int (cap3.get(4)) 
size1 = (frame1_width, frame1_height)
size2 = (frame2_width, frame2_height)
size3 = (frame3_width, frame3_height)
out11= cv2.VideoWriter('task4_output11.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, size1)
out22= cv2.VideoWriter('task4_output21.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, size2)
out33= cv2.VideoWriter('task4_output31.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10, size3)



#9. Main code for object detection implementation and top view transformaton

    #func1.forward(arg)

while True:
    print("w")
    ret1,frame1 = cap1.read()
    ret2,frame2 = cap2.read()
    ret3,frame3 = cap3.read()
    
    if not ret1 and not ret2 and not ret3:
        print("done")
        break
    
# # #10. Making sure each fram is valid
    if frame1 is not None and frame2 is not None and frame3 is not None:
        f1 = frame1
        f2=frame2
        f3 = frame3
       
        out11.write(frame1)
        out22.write(frame2)
        out33.write(frame3)

        height1,width1,channels1 = frame1.shape
        height2,width2,channels2 = frame2.shape
        height3,width3,channels3 = frame2.shape
#11. Preparing frames for feeding them into neural networks

    blob1 = cv2.dnn.blobFromImage(frame1,0.00392,(320,320),(0,0,0),True,crop=False)
    blob2 = cv2.dnn.blobFromImage(frame2,0.00392,(320,320),(0,0,0),True,crop=False)
    blob3 = cv2.dnn.blobFromImage(frame3,0.00392,(320,320),(0,0,0),True,crop=False)
    net1.setInput(blob1)
    net2.setInput(blob2)
    net3.setInput(blob3)
#12. Getting neural network predictions
    inps=[outputlayers1,outputlayers2,outputlayers3]
    start=time.time()
    outs=[]
    
    res=None
    outs1=net1.forward(inps[0])
    outs2=net2.forward(inps[1])
    outs3=net3.forward(inps[2])
    end=time.time()
  
    class_ids1=[]
    class_ids2=[]
    class_ids3=[]
    confidences1=[]
    confidences2=[]
    confidences3=[]
    boxes1=[]
    boxes2=[]
    boxes3=[]
    i = 0
    print("tot time",end-start)
    

    for out1 in outs1:
        for detection in out1:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #object detected
                center_x= int(detection[0]*width1)
                center_y= int(detection[1]*height1)
                w = int(detection[2]*width2)
                h = int(detection[3]*height2)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes1.append([x,y,w,h]) #put all rectangle areas
                confidences1.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids1.append(class_id) #name of the object tha was detected
    for out2 in outs2:
        for detection in out2:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #onject detected
                center_x= int(detection[0]*width2)
                center_y= int(detection[1]*height2)
                w = int(detection[2]*width2)
                h = int(detection[3]*height2)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes2.append([x,y,w,h]) #put all rectangle areas
                confidences2.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids2.append(class_id) #name of the object tha was detected
    for out3 in outs3:
        for detection in out3:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #onject detected
                center_x= int(detection[0]*width3)
                center_y= int(detection[1]*height3)
                w = int(detection[2]*width3)
                h = int(detection[3]*height3)
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                boxes3.append([x,y,w,h]) #put all rectangle areas
                confidences3.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids3.append(class_id) #name of the object tha was detected

    indexes1 = cv2.dnn.NMSBoxes(boxes1,confidences1,0.4,0.6)
    indexes2 = cv2.dnn.NMSBoxes(boxes2,confidences2,0.4,0.6)
    indexes3 = cv2.dnn.NMSBoxes(boxes3,confidences3,0.4,0.6)
    newb1 = boxes1
    newb2 = boxes2
    newb3 = boxes3
    
    view1 = cv2.warpPerspective(f1, H1, (567, 768))
    view2 = cv2.warpPerspective(f2, H2, (567, 768))
    view3 = cv2.warpPerspective(f3, H3, (567, 768))
    temp1=[]
    for i in range(len(boxes1)):
        if i in indexes1:
            x,y,w,h = boxes1[i]
            color = (0,0,255)
            ou=project_midpoint_into_top_view([[int(x+w/2)],[int(y+h/2)],[1]],H1)
            x1=int(ou[0]+0.5)
            pts=np.float32([[int(x+w/2)],[int(y+h/2)]]).reshape(-1,1,2)
            ouuu = cv2.perspectiveTransform(pts,H1)
            print("updated with perspective",ouuu)
            label = str(classes[class_ids1[i]])
            y1=int(ou[1]+0.5)
            obj = [x1,y1]
            if len(clicked_points) != 0:
                color = (0,0,255)
                cx = clicked_points[-1][0]
                cy = clicked_points[-1][1]
                rect = [cx,cy,cx+50,cy+50]
                detected =check_detection(obj,rect)
                print("obj1",obj,"rect1",rect)
                if detected:
                    # cv2.putText(view1,"detected1",(x1,y1+30),font,1,(255,255,255),2)
                    alert("alert msg", "!" + "ھوشیار"+"!"+"بندا ممنوعہ علاقہ میں موجود ہے")

            h1=3
            w1=3
            cv2.rectangle(view1,(x1,y1),(x1+w1,y1+h1),color,2)
            # cv2.putText(view1,label+" "+str(round(confidence,2)),(x1,y1+30),font,1,(255,255,255),2)
            
            confidence= confidences1[i]
            
            cv2.rectangle(frame1,(x,y),(x+w,y+h),color,2)
            # cv2.putText(frame1,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
    for i in range(len(boxes2)):    
        if i in indexes2:
            x,y,w,h = boxes2[i]
            color = (0,0,255)
            ou=project_midpoint_into_top_view([[int(x+w/2)],[int(y+h/2)],[1]],H2)
            pts=np.float32([[int(x+w/2)],[int(y+h/2)]]).reshape(-1,1,2)
            ouuu = cv2.perspectiveTransform(pts,H2)
            print("updated with perspective",ouuu)
            label = str(classes[class_ids2[i]])
            x1=int(ou[0]+0.5)
            y1=int(ou[1]+0.5)
            h1=3
            w1=3
            obj = [x1,y1]
            if len(clicked_points) != 0:
                color = (0,0,255)
                cx = clicked_points[-1][0]
                cy = clicked_points[-1][1]
                rect = [cx,cy,cx+50,cy+50]
                print("obj2",obj,"rect2",rect)
                detected =check_detection(obj,rect)
                if detected:
                    # cv2.putText(view2,"detected2",(x1,y1+30),font,1,(255,255,255),2)
                    alert("alert msg", "!" + "ھوشیار"+"!"+"بندا ممنوعہ علاقہ میں موجود ہے")

            cv2.rectangle(view2,(x1,y1),(x1+w1,y1+h1),color,2)
            # cv2.putText(view2,label+" "+str(round(confidence,2)),(x1,y1+30),font,1,(255,255,255),2)
            label = str(classes[class_ids2[i]])
            confidence= confidences2[i]
            
            cv2.rectangle(frame2,(x,y),(x+w,y+h),color,2)
            # cv2.putText(frame2,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
    for i in range(len(boxes3)): 
        if i in indexes3:
            color = (0,0,255)
            x,y,w,h = boxes3[i]
            ou=project_midpoint_into_top_view([[int(x+w/2)],[int(y+h/2)],[1]],H3)
            pts=np.float32([[int(x+w/2)],[int(y+h/2)]]).reshape(-1,1,2)
            ouuu = cv2.perspectiveTransform(pts,H3)
            print("updated with perspective",ouuu)
            label = str(classes[class_ids3[i]])
            x1=int(ou[0]+0.5)
            y1=int(ou[1]+0.5)
            print("view3 x ",x1,"y ",y1)
            h1=3
            w1=3
            obj = [x1,y1]
            if len(clicked_points) != 0:
                color = (0,0,255)
                cx = clicked_points[-1][0]
                cy = clicked_points[-1][1]
                rect = [cx,cy,cx+50,cy+50]
                print("obj3",obj,"rect3",rect)
                detected =check_detection(obj,rect)
                if detected:
                    # cv2.putText(view3,"detected3",(x1,y1+30),font,1,(255,255,255),2)
                    alert("alert msg", "!" + "ھوشیار"+"!"+"بندا ممنوعہ علاقہ میں موجود ہے")

            cv2.rectangle(view3,(x1,y1),(x1+w1,y1+h1),color,2)
            # cv2.putText(view3,label+" "+str(round(confidence,2)),(x1,y1+30),font,1,(255,255,255),2)
            label = str(classes[class_ids3[i]])
            confidence= confidences3[i] 
            cv2.rectangle(frame3,(x,y),(x+w,y+h),color,2)
            # cv2.putText(frame3,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
    

    k = cv2.waitKey(100) 
    if k == 27: #press Esc to exit
        break
    

    merged_view = merge_views(view1,view2,view3)
    cv2.namedWindow('merged_view')
    # mouse callback function

            
    cv2.setMouseCallback('merged_view',draw_rectangle)
    
    if len(clicked_points) != 0:
        color = (0,0,0)
        cx = clicked_points[-1][0]
        cy = clicked_points[-1][1]
        cv2.rectangle(merged_view,(cx,cy),(cx+50,cy+50),color,2)
    cv2.imshow('merged_view',merged_view )
    q = cv2.waitKey(1)
    if q == 27:
        break
save_boxes(boxes1,boxes2,boxes3)   
q = cv2.waitKey(0)            
cv2.destroyAllWindows()




