import subprocess, time, gc, os, sys

def setup_environment():
    try:
        ipy = get_ipython()
    except:
        ipy = 'could not get_ipython'
    
    if 'google.colab' in str(ipy):
        start_time = time.time()
        packages = [
            'https://download.pytorch.org/whl/cu118/xformers-0.0.22.post4%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl',
            'einops==0.4.1 pytorch-lightning==1.7.7 torchdiffeq==0.2.3 torchsde==0.2.5',
            'ftfy timm transformers open-clip-torch omegaconf torchmetrics==0.11.4',
            'safetensors kornia accelerate jsonmerge matplotlib resize-right',
            'scikit-learn numpngw pydantic',
            'fastapi nest-asyncio pyngrok uvicorn python-multipart',
            'openai==0.28 cohere tiktoken',
            'ffmpeg spleeter gdown'
        ]
        for package in packages:
            print(f"..installing {package}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + package.split())
        if not os.path.exists("deforum-stable-diffusion"):
            subprocess.check_call(['git', 'clone', '-b', '0.7.1', 'https://github.com/deforum-art/deforum-stable-diffusion.git'])
        else:
            print(f"..deforum-stable-diffusion already exists")
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend(['deforum-stable-diffusion/','deforum-stable-diffusion/src',])
        end_time = time.time()
        print(f"..environment set up in {end_time-start_time:.0f} seconds")
    else:
        sys.path.extend(['src'])
        print("..skipping setup")

def PathSetup():
    models_path = "models" #@param {type:"string"}
    configs_path = "configs" #@param {type:"string"}
    output_path = "outputs" #@param {type:"string"}
    mount_google_drive = True #@param {type:"boolean"}
    models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}
    return locals()

def ModelSetup():
    map_location = "cuda" #@param ["cpu", "cuda"]
    model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
    model_checkpoint =  "custom" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
    custom_config_path = "" #@param {type:"string"}
    custom_checkpoint_path = "models/model.ckpt" #@param {type:"string"}
    return locals()

def upscaleSetup():
    # Clone Real-ESRGAN
    subprocess.run(['git', 'clone', '-q', 'https://github.com/xinntao/Real-ESRGAN.git'])
    
    # Change directory to Real-ESRGAN
    os.chdir('Real-ESRGAN')
    
    # Set up the environment
    subprocess.run(['pip', 'install', 'basicsr', 'facexlib', 'gfpgan', '-q'])
    subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'])
    subprocess.run(['python', 'setup.py', 'develop'])
    
    # Download the pre-trained model
    subprocess.run(['wget', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', '-P', 'experiments/pretrained_models'])
    
    # Change back to the original directory
    os.chdir('..')

def downloadModel():
    import gdown
    subprocess.run(['mkdir', '-p', 'models'])
    gdown.download('https://drive.google.com/uc?export=download&id=12L9Z5fO748i8Ua9q0qZBhwwBJ0SyM0rg','models/model.ckpt')

def createPaths(path):
    img_path = os.path.join(path, 'images')
    samples_path = os.path.join(path, 'samples')
    upscale_path = os.path.join(img_path, 'results')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(samples_path, exist_ok=True)
    os.makedirs(upscale_path, exist_ok=True)


def create_key_value_object(arr, k):
    key_value_object = {n * k: value for n, value in enumerate(arr)}
    return key_value_object

def handle_strength_schedule(gap,length,default=0.65,pivot=0.5):
    result = f"0: ({default}), "
    for i in range(1,length):
        key_minus_1 = gap * i - 1
        key = gap * i
        key_plus_1 = gap * i + 1

        result += f"{key_minus_1}: ({default}), {key}: ({pivot}), {key_plus_1}: ({default}), "
    # Remove the trailing comma and space
    result = result.rstrip(", ")
    return result