import subprocess
import os

def upscale(path, model_name="RealESRGAN_x4plus", out_scale='2'):
    result_folder = os.path.join(path,'results')
    cmd = ["python", "Real-ESRGAN/inference_realesrgan.py", "-n", model_name, "-i", path,"-o",result_folder, "--outscale", out_scale]
    subprocess.run(cmd)