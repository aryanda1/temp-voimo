import os
from spleeter  import split
from whisperapi import saveLyrics
from types import SimpleNamespace
import time,gc, random
from createVideo import create_video
from keyFrame import keyFrames
from loadSetting import render_animation
from upscale import upscale as upsc
import torch
output_path = "output"
os.makedirs(output_path, exist_ok=True)
def musictoVid(device,model,mp3_path,theme,artist,option,upscale,timestring,api_key):
  from animSett import DeforumAnimArgs
  from loadSetting import DeforumArgs
  # print(f"output_path: {output_path}")
  split(mp3_path)
  fileName = mp3_path.split('/')[-1].split('.')[0]
  vocalsPath = output_path+'/'+fileName+'/vocals.wav'
  instrumentPath = output_path+'/'+fileName+'/accompaniment.wav'

  keyfr,maxFr = keyFrames(instrumentPath)
  if option<2:
    saveLyrics(vocalsPath,output_path,api_key)
  #   whisp.transcribe(vocalsPath)
  args_dict = DeforumArgs(output_path)
  anim_args_dict = DeforumAnimArgs(maxFr,keyfr)
  args = SimpleNamespace(**args_dict)
  anim_args = SimpleNamespace(**anim_args_dict)

  # args.timestring = time.strftime('%Y%m%d%H%M%S')
  args.timestring = timestring
  args.strength = max(0.0, min(1.0, args.strength))
  args.theme = theme
  args.artist = artist
  args.option = int(option)
  if args.seed == -1:
    args.seed = random.randint(0, 2**32 - 1)
  args.ddim_eta = 0

  # clean up unused memory
  gc.collect()
  torch.cuda.empty_cache()
  os.makedirs(os.path.join(args.outdir,'images'))
  render_animation(args, anim_args,device,model,api_key)
  if upscale:
    upsc(os.path.join(args.outdir,'images'), model_name="RealESRGAN_x4plus", out_scale='2')
  return create_video(args,anim_args,mp3_path,upscale)