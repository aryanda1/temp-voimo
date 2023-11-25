fps = 12
# from IPython import display
def create_video(args,anim_args):
    import os
    import subprocess
    image_path = os.path.join(args.outdir, (f"{args.timestring}_%05d.png") if not anim_args.upscale else os.path.join('results', f"{args.timestring}_%05d_out.png"))
    mp4_path = os.path.join(args.folder_path, f"{args.timestring}.mp4")
    mp3_path = os.path.join(args.folder_path,'audio.wav')
    print(image_path,' -> ',mp4_path)
    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-i',mp3_path,
        '-shortest',
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    return mp4_path