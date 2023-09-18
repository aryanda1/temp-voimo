import openai
import time
from tokenCount import num_tokens_from_messages

role = '''I want you to act as a Stable Diffusion Art Prompt Generator. The formula for a prompt is made of parts, the parts are indicated by brackets. The [Subject] is the person place or thing the image is focused on. [Emotions] is the emotional look the subject or scene might have. [Verb] is What the subject is doing, such as standing, jumping, working and other varied that match the subject. [Adjectives] like beautiful, rendered, realistic, tiny, colorful and other varied that match the subject. The [Environment] in which the subject is in, [Lighting] of the scene like moody, ambient, sunny, foggy and others that match the Environment and compliment the subject. [Photography type] like Polaroid, long exposure, GoPro, bokeh and others. Dont include monochrome, fisheye, and black and white photography. And [Quality] like High definition, 4K, 8K, 64K UHD, SDR and other. The subject and environment should match and have the most emphasis.
It is ok to omit one of the other formula parts. I will give you a [Subject], you will respond with a full prompt. Present the result as one full sentence, no line breaks, no delimiters, and keep it as concise as possible while still conveying a full scene.

Here is a sample of how it should be output: "Beautiful woman, contemplative and reflective, sitting on a bench, cozy sweater, autumn park with colorful leaves, soft overcast light, muted color photography style, 4K quality."
I Will give u input, u generate a prompts'''

role2 = '''I want you to act as a Stable Diffusion Art Prompt Generator. The formula for a prompt is made of parts, the parts are indicated by brackets. The [Subject] is the person place or thing the image is focused on. [Emotions] is the emotional look the subject or scene might have. [Verb] is What the subject is doing, such as standing, jumping, working and other varied that match the subject. [Adjectives] like beautiful, rendered, realistic, tiny, colorful and other varied that match the subject. The [Environment] in which the subject is in, [Lighting] of the scene like moody, ambient, sunny, foggy and others that match the Environment and compliment the subject. [Photography type] like Polaroid, long exposure, GoPro, bokeh and others. Dont include monochrome, fisheye, and black and white photography. And [Quality] like High definition, 4K, 8K, 64K UHD, SDR and other. The subject and environment should match and have the most emphasis.
It is ok to omit one of the other formula parts. I will give you a [Subject], you will respond with a full prompt. Present the result as one full sentence, no line breaks, no delimiters, and keep it as concise as possible while still conveying a full scene.

Here is a sample of how it should be output: "Beautiful woman, contemplative and reflective, sitting on a bench, cozy sweater, autumn park with colorful leaves, soft overcast light, muted color photography style, 4K quality."
I Will give u input, u generate 4 different aesthatic prompts'''

def gptResponse(role,mes):
  message=[
          {"role": "system", "content": role},
          {"role": "user", "content": mes},
      ]
  tokCnt = num_tokens_from_messages(message)
  res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    max_tokens=4000 - tokCnt,
    messages=message
  ) 
  return res['choices'][0]['message']['content']

def prompts(maxFrame,api_key,theme='',artists='',choice = 0):
  openai.api_key = api_key
  # from promptsSuf import suf
  
  animation_prompts = dict()
  frameGap = int(maxFrame/5)
  frame = 0
  if choice<2:
    with open("output/vocals.txt", 'r') as fp:
      x = (fp.readlines())
    # print(x)
    impWords = gptResponse('U take input as some sentences, and u Extract 12 important words from it. The output will be keywords separated by comma',x[0])
    newScenes = gptResponse('U take input as keywords, and create 5 different scenes from given input using max 5 words from the input, and only output the scene nothing extra. The scene should have minimum 50 characters',impWords)
    newScenes = "\n".join([sent for sent in newScenes.split('\n') if len(sent)>10])
    if newScenes[0][0]=='1':
        newScenes = "\n".join(['Scene '+str(idx)+': '+sent[3:] for idx,sent in enumerate(newScenes.split('\n'))])
    # print(newScenes)
    # print('kk')
    time.sleep(20)
    newKey = gptResponse('You are a keyword extraction assistant. Extract four keywords from each scene, ensuring the combined length of all keywords for each scene is under 100 characters.',newScenes)
    # print(newKey)
    for key in newKey.split('\n'):
        key = key.split(':')[-1]
        # print('Hello')
        time.sleep(20)
        # print('GG')
        # print(theme+','+key)
        prompt = gptResponse(role,theme+key)
        if len(prompt)>100:
            animation_prompts[frame] = prompt
            frame+=frameGap
  else:
    res2 = gptResponse(role2,theme)
    # print(res2)
    if res2[0]=='1':
      res2 = [sent[3:] for sent in res2.split('\n')]
    # print(res2)
    for prompt in res2:
      if(len(prompt)<40):
        continue
      animation_prompts[frame] = prompt
      frame+=frameGap

  # print(animation_prompts)
  if len(artists)>0:
    artistsPref = f'(style of {artists}) '
    for key, value in animation_prompts.items():
      animation_prompts[key] = artistsPref + value
  return animation_prompts
