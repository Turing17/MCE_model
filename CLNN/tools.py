flie = open('name_txt/name_data.txt', 'w+')
with open('name_txt/train_name.txt', 'r') as f:
    name_lines = f.readlines()
    for line in name_lines:
        flie.write(line.split()[0]+"_seg")
        flie.write('\n')
    f.close()
    flie.close()

