import os


def list_subdirectories(basedir,speakers,view):
    subdirectories = []
    for spk in speakers:
        spkpath = os.path.join(basedir, str(spk),str(view))
        for item in os.listdir(spkpath):
            item_path = os.path.join(spkpath, item)
            if os.path.isdir(item_path):
                # lipまでの部分を削除して相対パスを取得
                relative_path = os.path.relpath(item_path, start=base_directory)
                subdirectories.append(relative_path)
    return subdirectories

base_directory = './../lip/'




#数字だけ入力
val = [1,3,5,7,9]
view = 5

valout = []
trainout = []

for i in range(1,54):
    if i in val:
        valout.append(i)
    else:
        trainout.append(i)


if 29 in valout:
    valout.remove(29)

if 29 in trainout:
    trainout.remove(29)

# 出力するディレクトリのパスを取得
directories = list_subdirectories(base_directory, trainout,view)
print(directories)

# ファイルに書き込む
with open('./unseentrain.txt', 'w') as file:
    for item in directories:
        file.write(item + '\n')

directories = list_subdirectories(base_directory, valout,view)
with open('./unseenval.txt', 'w') as file:
    for item in directories:
        file.write(item + '\n')

