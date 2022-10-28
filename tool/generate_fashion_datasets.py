import os

# path for downloaded fashion images
root_fashion_images = '/home/deeplab/datasets/deepfashion/inshop/img_highres'

# import glob
# id2path = {}
# glob.glob(root_fashion_images + '/**/*.jpg', recursive=True)[:10]
# .replace('/home/deeplab/datasets/deepfashion/inshop/adgan/img','').replace('/','')

import re
# re.sub(r'(fashion)(WOMEN|MEN)(.+)(id)(\d{8})(\d{2}_\d)',r'/\2/\3/\4_\5/\6_','fashionWOMENSkirtsid0000062904_3back.jpg')

root_fashion_dir = '/home/deeplab/datasets/deepfashion/inshop/adgan/highres'
assert len(root_fashion_dir) > 0, 'please give the path of raw deep fashion dataset!'

train_images = []
train_f = open(os.path.join(root_fashion_dir,'train.lst'), 'r')
for lines in train_f:
	lines = lines.strip()
	if lines.endswith('.jpg'):
		train_images.append(lines)

test_images = []
test_f = open(os.path.join(root_fashion_dir,'test.lst'), 'r')
for lines in test_f:
	lines = lines.strip()
	if lines.endswith('.jpg'):
		test_images.append(lines)

train_path = os.path.join(root_fashion_dir,'train')
if not os.path.exists(train_path):
	os.mkdir(train_path)

for item in train_images:
	# from_ = os.path.join(root_fashion_dir, item)
	from_ = root_fashion_images + re.sub(r'(fashion)(WOMEN|MEN)(.+)(id)(\d{8})(\d{2}_\d)',r'/\2/\3/\4_\5/\6_',item)
	if not os.path.isfile(from_):
		print('not found:', from_)
	to_ = os.path.join(train_path, item)
	os.system('cp %s %s' %(from_, to_))


test_path = os.path.join(root_fashion_dir,'test')
if not os.path.exists(test_path):
	os.mkdir(test_path)

for item in test_images:
	# from_ = os.path.join(root_fashion_dir, item)
	from_ = root_fashion_images + re.sub(r'(fashion)(WOMEN|MEN)(.+)(id)(\d{8})(\d{2}_\d)',r'/\2/\3/\4_\5/\6_',item)
	if not os.path.isfile(from_):
		print('not found:', from_)
	to_ = os.path.join(test_path, item)
	os.system('cp %s %s' %(from_, to_))
