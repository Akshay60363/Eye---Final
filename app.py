from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
import pandas as pd
import numpy as np
import pymysql
import cv2
from PIL import Image, ImageTk
import face_recognition
import pickle
import sys
import pingenerate as pingen
import random as r
import pyotp
import smtplib 
from email.message import EmailMessage
from datetime import datetime
import train_faces as knntrain
import dlib
from scipy.spatial import distance as dist

# Compatibility shim for models pickled with older scikit-learn internals.
try:
    import sklearn.metrics._dist_metrics as _dist_metrics
    sys.modules.setdefault("sklearn.neighbors._dist_metrics", _dist_metrics)
    # sklearn >=1.3 uses suffixed class names (e.g., EuclideanDistance64),
    # while older pickles may reference unsuffixed names.
    for _name in dir(_dist_metrics):
        if _name.endswith("Distance64"):
            _legacy_name = _name[:-2]
            if not hasattr(_dist_metrics, _legacy_name):
                setattr(_dist_metrics, _legacy_name, getattr(_dist_metrics, _name))
except Exception:
    pass

print(cv2.__version__)
mydb = pymysql.connect(host='localhost', user='root', passwd='root', db='banking')
conn = mydb.cursor()
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
fontColor=(255,255,255)
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
clientid=0
otp="0000"
totp = pyotp.TOTP('base32secret3232')

@app.route('/')
#loading login page or main chat page
def index():
        if not session.get('logged_in'):
                return render_template("login.html")
        else:
                return render_template('dashboard.html')


@app.route('/registerpage',methods=['POST'])
def reg_page():
    return render_template("register.html")
@app.route('/forgetpasspage',methods=['POST'])
def fpass_page():
    return render_template("forgetpass.html")
        
@app.route('/loginpage',methods=['POST'])
def log_page():
    return render_template("login.html")
    

    
    
@app.route('/AddClient',methods=['POST'])
def main_page():
        cid=request.form['cid']
        name=request.form['name']
        fathername=request.form['fname']
        email=request.form['email']
        mob=request.form['mob']
        address=request.form['address']
        city=request.form['city']
        state=request.form['state']
        pnum=request.form['pnum']
        anum=request.form['anum']
        cmd="SELECT * FROM client WHERE cid='"+cid+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Client Id Already Exists")
                return render_template("dashboard.html",message="Client ID Already Exists")
        else:
                print("insert")
                cmd="INSERT INTO client Values('"+str(cid)+"','"+str(name)+"','"+str(fathername)+"','"+str(email)+"','"+str(mob)+"',"
                "'"+str(address)+"','"+str(city)+"','"+str(state)+"','"+str(pnum)+"','"+str(anum)+"')"
                print(cmd)
                print("Inserted Successfully")
                conn.execute(cmd)
                mydb.commit()
                return render_template("dashboard.html",message="Inserted SuccesFully")

@app.route('/UpdateClient',methods=['POST'])
def UpdateClient():
        cid=request.form['cid']
        name=request.form['name']
        fathername=request.form['fname']
        email=request.form['email']
        mob=request.form['mob']
        address=request.form['address']
        city=request.form['city']
        state=request.form['state']
        pnum=request.form['pnum']
        anum=request.form['anum']
        cmd="update client set cid='"+str(cid)+"',name='"+str(name)+"',fathername='"+str(fathername)+"',email='"+str(email)+"',"
        "mobile='"+str(mob)+"',address='"+str(address)+"',city='"+str(city)+"',state='"+str(state)+"',pnum='"+str(pnum)+"',anum='"+str(anum)
        +"' where cid='"+str(cid)+"'"
        print(cmd)
        conn.execute(cmd)
        mydb.commit()
        print("Update Successfully")
        return render_template("dashboard1.html",message="Update SuccesFully")
   

@app.route('/ViewClient',methods=['POST'])
def ViewClient():
        cid=request.form['cid']
        cmd="SELECT * FROM client WHERE cid='"+str(cid)+"'"
        conn.execute(cmd)
        cursor=conn.fetchall()
        print("length",len(cursor))
        if len(cursor)>0:
                results=[]
                for row in cursor:
                        print(row)
                results.append(row[0])
                results.append(row[1])
                results.append(row[2])
                results.append(row[3])
                results.append(row[4])
                results.append(row[5])
                results.append(row[6])
                results.append(row[7])
                results.append(row[8])
                results.append(row[9])
                print("length of row",len(row))
        
                return render_template("dashboard1.html",results=results)
        else:
                return render_template("dashboard1.html",message="No Records Found")
                

def StartCamera(cid,name):
        print("cid",cid)
        sampleNum=0
        cam = cv2.VideoCapture(0)
        sampleNum=0
        img_counter = 0
        DIR=f"./Dataset/{name}_{cid}"
        try:
                os.mkdir(DIR)
                print("Directory " , name ,  " Created ") 
        except FileExistsError:

                print("Directory " , name ,  " already exists")
                img_counter = len(os.listdir(DIR))
        while(True):
                ret, frame = cam.read()
                cv2.imshow("Video", frame)
                if not ret:
                        break
                k = cv2.waitKey(1)
                if k%256 == 27:
                        # ESC pressed
                        print("Escape hit, closing...")
                        break
                elif k%256 == 32:
                        # SPACE pressed
                        img_name = f"./Dataset/{name}_{cid}/opencv_frame_{img_counter}.png"
                        cv2.imwrite(img_name, frame)
                        print("{} written!".format(img_name))
                        img_counter += 1

        cam.release()
        cv2.destroyAllWindows()

def getProfile(Id):
        data=[]
        cmd="SELECT * FROM login WHERE cid="+str(Id)
        #print(cmd)
        conn.execute(cmd)

        cursor=conn.fetchall()
        #print(cursor)
        profile=None
        for row in cursor:
                #print(row)
                
                profile=row[1]

        print("data value",data)
        
        
        conn.close
        return profile
        
def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
        if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
                with open(model_path, 'rb') as f:
                        knn_clf = pickle.load(f)
        # Load image file and find face locations
        # X_img = face_recognition.load_image_file(X_img_path)
        X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)
        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
                return []
        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)
        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        # print(closest_distances)
        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]




def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

@app.route('/Detect', methods=['POST'])
def detection():
    print("detection")
    cam = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    blink_counter = 0
    total_blinks = 0
    blink_detected = False
    face_names = []
    process_this_frame = True
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # dlib detector expects 8-bit gray or RGB input.
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)

        if len(frame.shape) == 2:
            gray = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            shape = np.array(shape)
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                    blink_detected = True
                blink_counter = 0
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if not blink_detected:
            cv2.putText(frame, "Please blink to proceed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                predictions = predict(rgb_frame, model_path="./models/trained_model.clf")
            process_this_frame = not process_this_frame

            for name, (top, right, bottom, left) in predictions:
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Verified", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                if name.lower() != "unknown":
                    face_names.append(name)
                    break
        cv2.imshow('In Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or face_names:
            break
    cam.release()
    cv2.destroyAllWindows()
    if face_names:
        cmd = f"SELECT * FROM login WHERE name='{face_names[0]}'"
        conn.execute(cmd)
        cursor = conn.fetchall()
        results = [col for row in cursor for col in row]
        return render_template("login1.html", results=results)

@app.route('/Addpin',methods=['POST'])
def addpin():
        print("addpin")
        cid=request.form['cid']
        pin1=request.form['pin']
        results=[]
        
        pin=str(pin1)+pingen.process()
        print(pin)
        print("len",len(str(pin)))
        while len(pin)<4:
                pin=str(pin)+pingen.process()
                print(pin)
                #render_template("login.html",results=results,pin=pin)

        session['generated_pin'] = pin
        session['generated_cid'] = cid
        results.append(cid)
        results.append(pin)
        print(results)
        return render_template("login2.html",results=results)
    
@app.route('/register',methods=['POST'])
def reg():
        name=request.form['name']
        cid=request.form['cid']
        pin=request.form['pin']
        email=request.form['emailid']
        mobile=request.form['mobile']
        cmd="SELECT * FROM login WHERE cid='"+cid+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Username Already Exists")
                return render_template("login.html",message="Client id Already Exists")
        else:
                print("insert")
                cmd="INSERT INTO login Values('"+str(cid)+"','"+str(name)+"','"+str(pin)+"','"+str(email)+"','"+str(mobile)+"')"
                print(cmd)
                print("Inserted Successfully")
                conn.execute(cmd)
                mydb.commit()
                StartCamera(cid,name)
                knntrain.trainer()
                return render_template("login.html",message="Inserted SuccesFully")
        
@app.route('/changepin',methods=['POST'])
def upatepin():
        cid=request.form['cid']
        pin=request.form['pin']
        cpin=request.form['cpin']
        cmd="SELECT * FROM login WHERE cid='"+cid+"'"
        print(cmd)
        conn.execute(cmd)
        cursor=conn.fetchall()
        isRecordExist=0
        for row in cursor:
                isRecordExist=1
        if(isRecordExist==1):
                print("Username Already Exists")
                cmd="UPDATE login SET pin='"+str(pin)+"'WHERE cid='"+cid+"'"
                print(cmd)
                print("Inserted Successfully")
                
                conn.execute(cmd)
                mydb.commit()
                return render_template("login.html",message="PIN Updated")
                
        else:
                return render_template("forgetpass.html",message="Client id not Exist")

    
@app.route('/login1',methods=['POST'])
def log_in1():
        global otp
        otpvalue=request.form['otp']
        print(otp)
        print(otpvalue)
        if otp==otpvalue:
                session['logged_in'] = True
                session['cid'] = request.form['cid']
                return redirect(url_for('index'))
        else:
                return render_template("login.html",message="Check OTP Value")
@app.route('/fpass1',methods=['POST'])
def fpass_in1():
        global otp
        otpvalue=request.form['otp']
        print(otp)
        print(otpvalue)
        if otp==otpvalue:
                results=[]
                #session['logged_in'] = True
                session['cid'] = request.form['cid']
                cid=request.form['cid']
                results.append(cid)
                return render_template("changepin.html",results=results)
        else:
                return render_template("forgetpass.html",message="Check OTP Value")
                

    
@app.route('/login',methods=['POST'])
def log_in():
        global otp
        #complete login if name is not an empty string or doesnt corss with any names currently used across sessions
        if request.form['cid'] != None and request.form['cid'] != "" and request.form['pin'] != None and request.form['pin'] != "":
                cid=request.form['cid']
                pin=request.form['pin']
                generated_pin = session.get('generated_pin')
                generated_cid = session.get('generated_cid')
                using_generated_pin = (
                        generated_pin is not None and
                        generated_cid == cid and
                        pin == generated_pin
                )

                if using_generated_pin:
                        cmd="SELECT cid,pin,email FROM login WHERE cid='"+cid+"'"
                else:
                        cmd="SELECT cid,pin,email FROM login WHERE cid='"+cid+"' and pin='"+pin+"'"
                print(cmd)
                conn.execute(cmd)
                cursor=conn.fetchall()
                if len(cursor) > 0:
                        row = cursor[0]
                        mobile=row[2]
                        results=[]
                        results.append(cid)
                        results.append(pin)
                        print(otp)
                        otp=totp.now()
                        print(otp)
                        print("Email=",mobile)
                        msg = EmailMessage()
                        msg.set_content("Your OTP is : "+str(otp))
                        msg['Subject'] = 'OTP'
                        msg['From'] = "poisonousplants2024@gmail.com"
                        msg['To'] = mobile
                        s = smtplib.SMTP('smtp.gmail.com', 587)
                        s.starttls()
                        s.login("poisonousplants2024@gmail.com", "wtfghdcknihmbaog")
                        s.send_message(msg)
                        s.quit()

                        session.pop('generated_pin', None)
                        session.pop('generated_cid', None)
                        
                        return render_template("otp.html",results=results)
                else:
                        return render_template("login.html",message="Check Clinet id and Pin Number")

        return redirect(url_for('index'))

@app.route('/fpass',methods=['POST'])
def fpass_in():
        global otp
        #complete login if name is not an empty string or doesnt corss with any names currently used across sessions
        if request.form['name'] != None and request.form['name'] != "" and request.form['cid'] != None and request.form['cid'] != "":
                cid=request.form['name']
                pin=request.form['cid']
                cmd="SELECT cid,pin,email FROM login WHERE cid='"+pin+"'"
                print(cmd)
                conn.execute(cmd)
                cursor=conn.fetchall()
                isRecordExist=0
                for row in cursor:
                        isRecordExist=1
                        
                if(isRecordExist==1):
                        mobile=row[2]
                        results=[]
                        results.append(cid)
                        results.append(pin)
                        print(otp)
                        otp=totp.now()
                        print(otp)
                        print("Email=",mobile)
                        msg = EmailMessage()
                        msg.set_content("Your OTP is : "+str(otp))
                        msg['Subject'] = 'OTP'
                        msg['From'] = "poisonousplants2024@gmail.com"
                        msg['To'] = mobile
                        s = smtplib.SMTP('smtp.gmail.com', 587)
                        s.starttls()
                        s.login("poisonousplants2024@gmail.com", "wtfghdcknihmbaog")
                        s.send_message(msg)
                        s.quit()
                        return render_template("otp1.html",results=results)
                else:
                        return render_template("forgetpass.html",message="Check Clientid and Pin Number")

        return redirect(url_for('index'))
        
@app.route("/logout")
def log_out():
    session.clear()
    return redirect(url_for('index'))
    
@app.route("/AddClient")
def addClient():
    return render_template("dashboard.html")

@app.route("/TrainingPage")
def TrainingPage():
    return render_template("Training.html")
   
@app.route("/ViewClientPage")
def ViewClientPage():
    return render_template("dashboard1.html")
  
@app.route("/ReportPage")
def ReportPage():
        sql="select * from client"
        conn.execute(sql)
        results = conn.fetchall()
        return render_template("dashboard2.html",result=results)  
        
    
# /////////socket io config ///////////////
#when message is recieved from the client    
@socketio.on('message')
def handleMessage(msg):
    print("Message recieved: " + msg)
 
# socket-io error handling
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass
  
if __name__ == '__main__':
    socketio.run(app,debug=True,host='127.0.0.1', port=4000)
