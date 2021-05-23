from __future__ import print_function
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import imutils
import numpy as np
import datetime
import dlib
from scipy.spatial import distance as dist
import scipy.ndimage.filters as signal
from imutils import face_utils
import datetime
import imutils
import dlib
import matplotlib.pyplot as plt
import json
import codecs
# import tkinter as tk
# from tkinter import *
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage.interpolation import shift
import pickle
from queue import Queue
# import the necessary packages
import numpy as np
import cv2
#import twilio for text/call
# from twilio.rest import Client
# client = Client('AC70ff03021de6e57806ce0912d513db66','f495894474109fd17ccbb79145680e4b')
               
# inference
import drowsiness_stable.Infer as Infer
from collections import deque
import cv2
from aiohttp import web
from av import VideoFrame
from queue import Queue
from collections import deque

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

FRAME_MARGIN_BTW_2BLINKS=3
MIN_AMPLITUDE=0.04
MOUTH_AR_THRESH=0.35
MOUTH_AR_THRESH_ALERT=0.30
MOUTH_AR_CONSEC_FRAMES=20

class Blink():
    def __init__(self):

        self.start=0 #frame
        self.startEAR=1
        self.peak=0  #frame
        self.peakEAR = 1
        self.end=0   #frame
        self.endEAR=0
        self.amplitude=(self.startEAR+self.endEAR-2*self.peakEAR)/2
        self.duration = self.end-self.start+1
        self.EAR_of_FOI=0 #FrameOfInterest
        self.values=[]
        self.velocity=0  #Eye-closing velocity
    


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"


    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
   
    def mouth_aspect_ratio(self, mouth):

        A = dist.euclidean(mouth[14], mouth[18])

        C = dist.euclidean(mouth[12], mouth[16])

        if C<0.1:           #practical finetuning
            mar=0.2
        else:
            # compute the mouth aspect ratio
            mar = (A ) / (C)

        # return the mouth aspect ratio
        return mar

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])

        if C<0.1:           #practical finetuning due to possible numerical issue as a result of optical flow
            ear=0.3
        else:
            # compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)
        if ear>0.45:        #practical finetuning due to possible numerical issue as a result of optical flow
            ear=0.45
        # return the eye aspect ratio
        return ear

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.reference_frame = 0
        self.Q = Queue(maxsize=7)
        self.start = datetime.datetime.now()
        self.deque_blinks = deque(maxlen=30)
        self.detector = dlib.get_frontal_face_detector()
        #Load the Facial Landmark Detector
        self.predictor = dlib.shape_predictor('./drowsiness_stable/shape_predictor_68_face_landmarks.dat')
        #Load the Blink Detector
        self.loaded_svm = pickle.load(open('./drowsiness_stable/Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav', 'rb'))
        self.number_of_frames =0
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.COUNTER = 0
        self.MCOUNTER=0
        self.TOTAL = 0
        self.MTOTAL=0
        self.TOTAL_BLINKS=0
        self.Counter4blinks=0
        self.skip=False # to make sure a blink is not counted twice in the Blink_Tracker function
        self.Last_Blink=Blink()
        self.EAR_series=np.zeros([13])


    async def recv(self):
        predictor = self.predictor
        detector = self.detector
        loaded_svm = self.loaded_svm
        frame = await self.track.recv()
        number_of_frames = self.number_of_frames
        (lStart,lEnd) = (self.lStart, self.lEnd)
        (rStart, rEnd) = (self.rStart, self.rEnd)
        (mStart, mEnd) = (self.mStart, self.mEnd)
        MCOUNTER = self.MCOUNTER
        COUNTER = self.COUNTER
        TOTAL = self.TOTAL
        MTOTAL = self.MTOTAL
        TOTAL_BLINKS = self.TOTAL_BLINKS
        Counter4blinks = self.Counter4blinks
        skip = self.skip
        Last_Blink = self.Last_Blink
        EAR_series = self.EAR_series
        Q = self.Q
       
        if self.transform == "cartoon":
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            
            img = frame.to_ndarray(format="bgr24")
            new_frame = imutils.resize(img, width=450)
            gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)   #Brighten the image(Gamma correction)
            self.reference_frame = self.reference_frame + 1
            gray=self.adjust_gamma(gray,gamma=1.5)
            Q.put(new_frame)
            end = datetime.datetime.now()
            start = self.start
            ElapsedTime = (end-start).total_seconds()
            
            rects = detector(gray, 0)
            print(rects)
            if (np.size(rects) != 0):
                number_of_frames = number_of_frames + 1  # we only consider frames that face is detected
                First_frame = False
                old_gray = gray.copy()
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                Mouth = shape[mStart:mEnd]
                MAR = self.mouth_aspect_ratio(Mouth)
                MouthHull = cv2.convexHull(Mouth)
                #cv2.drawContours(frame, [MouthHull], -1, (255, 0, 0), 1)

                if MAR > MOUTH_AR_THRESH:
                    MCOUNTER += 1

                elif MAR < MOUTH_AR_THRESH_ALERT:
                    if MCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        MTOTAL += 1
                    MCOUNTER = 0


                ##############YAWNING####################
                #########################################

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR =  self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                #EAR_series[reference_frame]=ear
                EAR_series = shift(EAR_series, -1, cval=ear)

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                #cv2.drawContou

                print(self.reference_frame)
                print(Q.full())
           
         
         
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))
        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
