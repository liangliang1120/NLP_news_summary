import os

def merge_corpus():
    '''
    合并分词后的文件
    '''
    output = open("./wiki_cut_merge_r3.txt", "w", encoding="utf-8")
    input = "./"
    for i in range(1,4):
        n = str(i)
        s = n.zfill(3)
        print('is merging wiki_cut/wiki_{}.txt'.format(s))
        file_path = os.path.join(input, "wiki_cut/wiki_{}.txt".format(s))
        file = open(file_path, "r", encoding="utf-8")
        line = file.readline()
        while line:
            output.writelines(line)
            line = file.readline()
        file.close()
    output.close()

if __name__ == "__main__":
    merge_corpus()