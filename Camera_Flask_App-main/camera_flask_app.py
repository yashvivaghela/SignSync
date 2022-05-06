from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

DIR_PATH = 'D:\\Flask-Camera\\Camera_Flask_App-main'

WORKSPACE_PATH = DIR_PATH + '\\Tensorflow\\workspace'
SCRIPTS_PATH = DIR_PATH + '\\Tensorflow\\scripts'
APIMODEL_PATH = DIR_PATH + '\\Tensorflow\\models'
ANNOTATION_PATH = WORKSPACE_PATH+'\\annotations'
IMAGE_PATH = WORKSPACE_PATH+'\\images'
MODEL_PATH = WORKSPACE_PATH+'\\models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'\\pre-trained-models'
CONFIG_PATH = MODEL_PATH+'\\my_ssd_mobnet\\pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'\\my_ssd_mobnet\\'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

# CONFIG_PATH = MODEL_PATH+'\\'+CUSTOM_MODEL_NAME+'\\pipeline.config'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'\\label_map.pbtxt')



global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    # global net
    # (h, w) = frame.shape[:2]
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
    #     (300, 300), (104.0, 177.0, 123.0))   
    # net.setInput(blob)
    # detections = net.forward()
    # confidence = detections[0, 0, 0, 2]

    # if confidence < 0.5:            
    #         return frame           

    # box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    # (startX, startY, endX, endY) = box.astype("int")
    # try:
    #     frame=frame[startY:endY, startX:endX]
    #     (h, w) = frame.shape[:2]
    #     r = 480 / float(h)
    #     dim = ( int(w * r), 480)
    #     frame=cv2.resize(frame,dim)
    # except Exception as e:
    #     pass
    # while True: 
    # ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=.9,
                agnostic_mode=False)

    # cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    frame = cv2.resize(image_np_with_detections, (800, 600))
    # frame = cv2.flip(frame, 1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     # cap.release()
        #     break

    return frame
 

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        # frame = cv2.flip(frame, 1)

        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                # ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

# def draw_function(event, x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         global b,g,r,xpos,ypos, clicked
#         clicked = True
#         xpos = x
#         ypos = y
#         b,g,r = img[y,x]
#         b = int(b)
#         g = int(g)
#         r = int(r)
#         print("xpos: "+xpos)
#         print("ypos: "+ypos)
# cv2.setMouseCallback('image',draw_function)        


@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_color', methods=['GET', 'POST'])
def detect_color():
    if request.method == 'GET':
        return render_template('color_detection.html')
    elif request.method == 'POST':
        img_path = request.form.get('og_img')
        if img_path != '':
            os.system(f'python color_detection.py -i {img_path}')
        else:
            return render_template('color_detection.html')
    return render_template('color_detection.html')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Only':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     