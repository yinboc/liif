import os
from PIL import Image
from tqdm import tqdm

for size in [256, 128, 64, 32]:
    if size == 256:
        inp = './data1024x1024'
    else:
        inp = './256'
    print(size)
    os.mkdir(str(size))
    filenames = os.listdir(inp)
    for filename in tqdm(filenames):
        Image.open(os.path.join(inp, filename)) \
            .resize((size, size), Image.BICUBIC) \
            .save(os.path.join('.', str(size), filename.split('.')[0] + '.png'))
