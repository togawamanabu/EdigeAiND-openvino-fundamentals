"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
from collections import deque

logging.basicConfig(level=logging.DEBUG)


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
TOPIC = 'person'


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    
    single_image_mode = False
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    frame_count = 0
    FPS = 10
    count_stblizer = deque(maxlen=FPS)

    ### TODO: Load the model through `infer_network` ###
    ifvalues  = infer_network.load_model(model=args.model, device=args.device, cpu_extension=args.cpu_extension)
    
    n, c, h, w = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        ## LOAD A SINGLE IMAGE TO INFERENCE
        input_stream = args.input
        single_image_mode = True
    else:
        input_stream = args.input

    #THIS CODE LOAD SINGLE IMAGE OR VIDEO OR CAMRRA STREAM
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    
    if not cap.isOpened():
        logging.error("Unable to open video source")
        
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        
        if not flag:
            break
            
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###        
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape(n, c, h, w)

        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        infer_network.exec_net(cur_request_id, image)
        
        ### TODO: Wait for the result ###
        if infer_network.wait() ==0:
            det_time = time.time() - inf_start

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            out_frame, current_count = ssd_out(frame, result, prob_threshold, infer_network.person_id, initial_w, initial_h )                        
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            #logging.debug(out_frame.shape)

            ### TODO: Calculate and send relevant information on ###            
            ### current_count, total_count and duration to the MQTT server ###                     
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###      
            
            ## stablize count
            count_stblizer.append(current_count)
            
            if FPS <= len(count_stblizer):
                ## most offen number in frames
                mon = np.argmax(np.bincount(count_stblizer))
                
                current_count = int(mon)
                                    
            if current_count > last_count:
                frame_count = 0
                total_count = total_count + current_count - last_count

                client.publish("person", json.dumps({"total":total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = frame_count / FPS
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                
            client.publish("person", json.dumps({"count": current_count}))
            
            last_count = current_count
            frame_count += 1
            
            if key_pressed == 27:
                break

            ### TODO: Send the frame to the FFMPEG server ###
            sys.stdout.buffer.write(out_frame)
            sys.stdout.flush()

            ### TODO: Write an output image if `single_image_mode` ###
            if single_image_mode:
                cv2.imwrite('output_image.jpg', frame)
            
    
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()

def ssd_out(frame, result, prob_threshold, person_id, initial_w, initial_h):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        
        #logging.debug(obj[2])
        
        if obj[1] == person_id and obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count
        
        
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
