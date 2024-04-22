#*****************************************************************************
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#*****************************************************************************

import gradio as gr
import argparse
from streamer import OvmsStreamer

parser = argparse.ArgumentParser(description='Gradio frontend launcher')

parser.add_argument('--web_url',
                    required=True,
                    help='Web server URL')
parser.add_argument('--ovms_url',
                    required=True,
                    help='OVMS server URL')
args = parser.parse_args()


video_to_play = None

def callback(message, history):
    global video_to_play
    streamer = OvmsStreamer(args.ovms_url.split(':')[0], int(args.ovms_url.split(':')[1]))
    streamer.request_async(message)
    result = ""
    videofile = ""
    compflag = False
    for completion in streamer:
        #print(completion, end='', flush=True)
        if compflag == True:
            videofile += completion
        if completion != '#' and compflag == False:
            result += completion
        else:
            compflag = True        
        yield result
    videofile=videofile[:-3]
    video_to_play = "documents/videos/" + videofile
    print(result, flush=True)
    print(videofile, flush=True)
    

def vcallback(video):
    print(video_to_play)
    return video_to_play
    
with gr.Blocks() as demo:
    with gr.Row():        
        with gr.Column(scale=1, min_width=200):
            ChatBlock = gr.ChatInterface(callback, retry_btn=None, undo_btn=None) 
        VidBlock = gr.Interface(fn=vcallback, allow_flagging="never", inputs=None, outputs=gr.Video(None,  interactive=False, scale=4, autoplay=True, show_download_button=False, show_share_button=False))
 

demo.launch(server_name=args.web_url.split(':')[0], server_port=int(args.web_url.split(':')[1]))
