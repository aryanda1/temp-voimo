import subprocess
import os
import shutil

result_folder = os.path.join('/content/results')

def upscale(path, model_name="RealESRGAN_x4plus", out_scale='2'):
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.mkdir(result_folder)
    cmd = ["python", "Real-ESRGAN/inference_realesrgan.py", "-n", model_name, "-i", path, "--outscale", out_scale]
    subprocess.run(cmd)