setup_environment = True 
print_subprocess = False

if setup_environment:
    import subprocess, time
    print("Setting up environment...")
    start_time = time.time()
    all_process = [
        ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
        ['pip', 'install', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'],
        ['git', 'clone', 'https://github.com/deforum/stable-diffusion'],
        ['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers'],
        ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip'],
        ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq'],
        ['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git'],
        ['git', 'clone', 'https://github.com/isl-org/MiDaS.git'],
        ['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git'],
        ['apt', 'install', 'ffmpeg'],
        ['pip', 'install', 'spleeter'],
        ['pip','install','openai'],
        ['pip','install','fastapi','nest-asyncio','pyngrok','uvicorn'],
        ['pip','install','python-multipart'],
        ['pip','install','tiktoken'],
        ['pip','install','gdown'],
    ]
    for process in all_process:
        running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
        if print_subprocess:
            print(running)
    
    print(subprocess.run(['git', 'clone', 'https://github.com/deforum/k-diffusion/'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    with open('k-diffusion/k_diffusion/__init__.py', 'w') as f:
        f.write('')

    end_time = time.time()
    print(f"Environment set up in {end_time-start_time:.0f} seconds")