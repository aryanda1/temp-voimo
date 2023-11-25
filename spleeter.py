#A python file to split a wav file into vocals and other instruments
import subprocess
def split(input_path,output_path):
    cmd = ['spleeter','separate'
        ,'-p','spleeter:2stems',
        '-o',output_path,
        input_path]
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout,stderr = process.communicate()
    print(stdout)
    print(stderr)