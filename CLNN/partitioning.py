import os
import random
from tqdm import tqdm

# with open('./bin/dataset_name.txt', 'r') as f:
def add_p(name):
    temp = []
    p = ['flair', 't1ce', 't2']
    for i in p:
        temp.append(name.split()[0]+'_'+i+'\n')
    return temp

with open('name_txt/all_ids.txt', 'r') as f:

    list_dataset = f.readlines()
# path_save = r'D:\Desktop\DRN-model\bin\nametxt\brats2018'
path_save = r''
num = len(list_dataset)
txt_name_train = open(os.path.join(path_save, "name_txt/train_name.txt"), "w")
txt_name_val = open(os.path.join(path_save, "name_txt/val_name.txt"), "w")
list_train = random.sample(list_dataset, int(num*0.8)-1)
print(len(list_train))
for name in tqdm(list_dataset):
    if name in list_train:
        temp = add_p(name)
        for i in temp:
            txt_name_train.write(i)
            # txt_name_train.write("\n")
    else:
        temp = add_p(name)
        for i in temp:
            txt_name_val.write(i)
            # txt_name_val.write("\n")
txt_name_train.close()
txt_name_val.close()

# import os
# import random
# from PIL import Image, TarIO
# from torch.fx.experimental.accelerator_partitioner import Partitioner
# from torchvision import transforms
# import tarfile
# from setting import parse_opts
# from tqdm import tqdm
# import numpy as np
#
# # train = 256(H:180,L:76),val = 64(H:45,L:19) , test = 49(H=34,L=15),all=369
# # train = 304(H:214,L:90),val = 64(H:44,L:20) , test = 0(H=0,L=0),all=368
# sets = parse_opts()
# random.seed(sets.random_seed)
# print("random_seed:", sets.random_seed)
#
#
# class Partitioner_brats(object):
#     def __init__(self, path_save=None, path_dataset_brats=None):
#         self.path_save = path_save  # ./bin/nametxt/brats
#         self.path_dataset_brats = path_dataset_brats
#
#     def get_list_of_brats_name(self, num_train):
#         list_names_brats = os.listdir(self.path_dataset_brats)
#         list_names_train = random.sample(list_names_brats, num_train)
#         return list_names_brats, list_names_train
#
#     def get_txt_name(self, num_train):
#         if not os.path.exists(self.path_save):
#             os.makedirs(self.path_save)
#         txt_brats_train = open(os.path.join(self.path_save, "train_name.txt"), "w")
#         txt_brats_val = open(os.path.join(self.path_save, "val_name.txt"), "w")
#         txt_brats_test = open(os.path.join(self.path_save, "test_name.txt"), "w")
#         list_names_brats, list_names_train = self.get_list_of_brats_name(num_train)
#         for name in tqdm(list_names_brats):
#             if name in list_names_train:
#                 name = name.split('_flair')[0]
#                 txt_brats_train.write(name)
#                 txt_brats_train.write("\n")
#             else:
#                 name = name.split('_flair')[0]
#                 txt_brats_val.write(name)
#                 txt_brats_val.write("\n")
#         txt_brats_train.close()
#         txt_brats_val.close()
#         txt_brats_test.close()
#
#         print('Partitioning finish!')
#
#
# class Partitioner_ImageNet():
#     def tensor_sour_load(self, name_member, out_type='Image'):
#         fp = TarIO.TarIO(path_sour + '{}.tar'.format(name_member.split('_')[0]), name_member)
#         img_sour = Image.open(fp)
#         if out_type == 'ndarray':
#             img = np.asarray(img_sour)
#             img.flags.writeable = True
#         transforms_sour = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#         tensor_sour = transforms_sour(img_sour)
#         return tensor_sour
#
#     def get_name_txt_imagenet(self):
#         txt_imagenet_name = open(os.path.join(path_save, "imagenet_name.txt"), "w")
#         list_class_imagenet = os.listdir(path_sour)
#         for class_imagenet in tqdm(list_class_imagenet):
#             index = 0
#             path_c_imagenet = os.path.join(path_sour, class_imagenet)
#             with tarfile.open(path_c_imagenet) as tar:
#                 for member in tar.getmembers():
#                     if self.tensor_sour_load(member.name).shape[0] == 3:
#                         txt_imagenet_name.write(member.name)
#                         txt_imagenet_name.write("\n")
#                         index += 1
#                     if index == 100:
#                         break
#         txt_imagenet_name.close()
#
#     def get_train_val_name_imagenet(self):
#         txt_imagenet_train = open(os.path.join(path_save, "train_name.txt"), "w")
#         txt_imagenet_val = open(os.path.join(path_save, "val_name.txt"), "w")
#         with open(os.path.join(path_save, "imagenet_name.txt"), 'r') as f:
#             name_lines = f.readlines()
#         for i in tqdm(range(100)):
#             temp_lines = random.sample(name_lines[i * 100:(i + 1) * 100], 80)
#             for name in name_lines[i * 100:(i + 1) * 100]:
#                 if name in temp_lines:
#                     txt_imagenet_train.write(name.split()[0])
#                     txt_imagenet_train.write('\n')
#                 else:
#                     txt_imagenet_val.write(name.split()[0])
#                     txt_imagenet_val.write('\n')
#         txt_imagenet_val.close()
#         txt_imagenet_train.close()
#
#
# class Partitioner_Cat_Dog():
#     def get_name_txt_imagenet(self, path_save):
#         if os.path.exists(path_save) is False:
#             os.makedirs(path_save)
#         txt_name_train = open(os.path.join(path_save, "train_name.txt"), "w")
#         txt_name_val = open(os.path.join(path_save, "val_name.txt"), "w")
#         list_dataset = os.listdir(path_sour)
#         list_train = random.sample(list_dataset, int(len(list_dataset) * 0.7))
#         for name in tqdm(list_dataset):
#             if name in list_train:
#                 txt_name_train.write(name)
#                 txt_name_train.write("\n")
#             else:
#                 txt_name_val.write(name)
#                 txt_name_val.write("\n")
#         txt_name_train.close()
#         txt_name_val.close()
#
#
# class Partitioner_brats_2D_label14:
#     def __init__(self):
#         pass
#
#     def get_train_val_txt(self,list_train_brats, list_id_slicer, path_save):
#         '''
#
#         :param list_train_brats: 样本划分
#         :param list_id_slicer: 切片id
#         :param path_save:
#         :return:
#         '''
#         if not os.path.exists(path_save):
#             os.makedirs(path_save)
#         txt_brats_train_slicer = open(os.path.join(path_save, "val_all_slicer_name.txt"), "w")
#         for slicer_id in list_id_slicer:
#             if slicer_id.split(' ')[0]+'\n' in list_train_brats:
#                 txt_brats_train_slicer.write(slicer_id)
#                 # txt_brats_train_slicer.write('\n')
#         txt_brats_train_slicer.close()
#     def get_train_txt(self,list_train_slicer, path_save):
#         txt = open(os.path.join(path_save,'train.txt'),'w+')
#         for slicer_id in list_train_slicer:
#             slicer_i =slicer_id.split('\n')[0]
#             for i in slicer_i.split(' '):
#                 temp = slicer_i.split(' ')[0]+' {}'.format(i)
#                 if "BraTS20_Training" not in i:
#                     txt.write(temp)
#                     txt.write('\n')
#         txt.close()
#     def get_val_txt(self,list_val_slicer, path_save):
#         txt = open(os.path.join(path_save,'val.txt'),'w+')
#         for slicer_id in list_val_slicer:
#             slicer_i =slicer_id.split('\n')[0]
#             temp = slicer_i.split(' ')[0]+' {}'.format(slicer_i.split(' ')[1])
#             txt.write(temp)
#             txt.write('\n')
#
#         txt.close()
#
#
#
# if __name__ == "__main__":
#     path_save = './bin/nametxt/brats_2d_label14'
#     path_data = r"G:/ILSVRC2012_img_train"
#     path_obj = './bin/brats_npy/brats2020_A_2D/flair'
#     path_sour = r'./bin/dogs-vs-cats/train'
#     path_txt_obj_name = './bin/nametxt/brats/val_name.txt'
#     path_txt_sour_name = './bin/nametxt/imagenet/train_name.txt'
#     path_label_name = './bin/nametxt/imagenet/label_imagenet.txt'
#     path_txt_id_slicer = '/media/ps/Xueh/brats_2d_label14/slicer_id.txt'
#     path_txt_train_slicer_id = './bin/nametxt/brats_2d_label14/train_all_slicer_name.txt'
#     path_txt_val_slicer_id = './bin/nametxt/brats_2d_label14/val_all_slicer_name.txt'



