import os,glob,copy
from spleeter  import split
from whisperapi import saveLyrics
from types import SimpleNamespace
import gc
from createVideo import create_video
from keyFrame import keyFrames
import torch
import clip
from upscale import upscale as upsc
from prompts import prompts
from animSett import DeforumAnimArgs,DeforumArgs
from helpers.render import render_animation
from helpers.aesthetics import load_aesthetics_model
from helpers.prompts import Prompts
import random
import subprocess
from modelSetup import create_key_value_object,handle_strength_schedule
class VideoCreator:
  root = None
  api_key = None
  def __init__(self,output_path,theme,artist,option,upscale,timestring) -> None:
    img_path = os.path.join(output_path,'images')
    mp3_path = os.path.join(output_path,'audio.wav')
    split(mp3_path,output_path)
    fileName = mp3_path.split('/')[-1].split('.')[0]
    vocalsPath = output_path+'/'+fileName+'/vocals.wav'
    instrumentPath = output_path+'/'+fileName+'/accompaniment.wav'
    
    keyfr,maxFr = keyFrames(instrumentPath)
    if option<2:
      saveLyrics(vocalsPath,output_path,self.api_key)
    #   whisp.transcribe(vocalsPath)
    args_dict = DeforumArgs(img_path)
    anim_args_dict = DeforumAnimArgs(maxFr,keyfr)
    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)

    # args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.folder_path = output_path
    args.timestring = timestring
    args.strength = max(0.0, min(1.0, args.strength))
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
      self.root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(self.root.device)
      if (args.aesthetics_scale > 0):
          self.root.aesthetics_model = load_aesthetics_model(args, self.root)
    if not args.use_init:
      args.init_image = None
    args.ddim_eta = 0
    anim_args.upscale = upscale
    print('Creating prompts')
    anim_args.prompts = prompts(self.api_key,output_path,theme,artist,int(option))
    self.args = args
    self.anim_args = anim_args

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

  def create_video_from_music(self):
    self.anim_args.max_frames = self.anim_args.tot_frames
    self.args.seed_iter_N = 1
    self.args.outdir = os.path.join(self.args.folder_path,'images')
    # subprocess.run(['rm','-rf',self.args.outdir+'/*'])
    self.args.strength=0.65
    frame_gap = self.anim_args.tot_frames//len(self.anim_args.prompts)
    self.anim_args.strength_schedule=handle_strength_schedule(frame_gap,len(self.anim_args.prompts))
    prompt = create_key_value_object(self.anim_args.prompts,frame_gap)
    cond, uncond = Prompts(prompt=prompt,neg_prompt={0:''}).as_dict()
    render_animation(self.root, copy.deepcopy(self.anim_args),copy.deepcopy(self.args),cond,uncond)
    if self.anim_args.upscale:
      upsc(self.args.outdir, model_name="RealESRGAN_x4plus", out_scale='2')
    return create_video(self.args,self.anim_args)
  
  def gen_samples(self):
    self.anim_args.max_frames = len(self.anim_args.prompts)
    frame_gap = self.anim_args.tot_frames//len(self.anim_args.prompts)
    self.args.seed_iter_N = frame_gap
    self.args.seed = random.randint(0, 2**32 - 1)
    self.args.outdir = os.path.join(self.args.folder_path,'samples')
    print(self.args.outdir)
    # subprocess.run(['rm','-rf',self.args.outdir+'/*'])
    files = glob.glob(f'{self.args.outdir}/*')
    for f in files:
        os.remove(f)
    prompt = create_key_value_object(self.anim_args.prompts,1)
    cond, uncond = Prompts(prompt=prompt,neg_prompt={0:''}).as_dict()
    # print('Gen frames')
    # print(prompt)
    # print(self.root)
    # print(self.anim_args)
    # print(self.args)
    # print(cond,uncond)
    render_animation(self.root, copy.deepcopy(self.anim_args),copy.deepcopy(self.args),cond,uncond)