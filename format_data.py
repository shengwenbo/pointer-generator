# -*-coding:utf-8-*-
import os
import struct
import collections
import jieba.posseg as seg
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

origin_data = "./datas/origin/data.train.seg"
train_file = "./datas/origin/train.txt"
val_file = "./datas/origin/val.txt"
test_file = "./datas/origin/test.txt"
finished_files_dir = "./datas/new"

TRAIN_SIZE = 500000
VAL_SIZE = 10000
TEST_SIZE = 1000

VOCAB_SIZE = 30000

def cut(path):
    with open(path, encoding="utf-8") as fin, open(path+".seg","w",encoding="utf-8") as fout:
        out_lines = []
        for line in fin.readlines():
            segs = []
            for sub_line in line.strip().split("\t"):
                for token in seg.cut(sub_line):
                    word = token.word
                    tag = token.flag
                    segs.append("%s_%s" % (word, tag))
                    # if len(word) == 1:
                    #     segs.append("%s/%s_S" % (word, tag))
                    # else:
                    #     segs.append("%s/%s_B" % (word[0], tag))
                    #     [segs.append("%s/%s_M" % (c, tag)) for c in word[1:len(word)-1]]
                    #     segs.append("%s/%s_E" % (word[-1], tag))
                segs.append("\t")
            segs.append("\n")
            out_lines.append(" ".join(segs))
        fout.writelines(out_lines)


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines

def write_to_bin(input_file, out_file, makevocab=False, segment=False, repeat=1):
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        # read the  input text file , make even line become article and odd line to be abstract（line number begin with 0）
        lines = read_text_file(input_file)

        for line in lines:
            [_, _, content] = line.strip().split("\t", 2);
            if content.find("\t") >= 0:
                [origin, corrects] = content.split("\t", 1)
                if segment:
                    origin = origin.strip()
                else:
                    origin = " ".join(list(origin))
                origin = " ".join([origin for _ in range(repeat)])
                if corrects:
                    corrects = corrects.split("\t")
                    for c in corrects:
                        if segment:
                            c = c.strip()
                        else:
                            c = " ".join(list(c))
                        write_sent_pairs(origin, c, writer)
                        # write_sent_pairs(c, c, writer)
                # else:
                #     write_sent_pairs(origin, origin, writer)

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = origin.split(' ')
                    tokens = art_tokens
                    tokens = [t.strip() for t in tokens]  # strip
                    tokens = [t for t in tokens if t != "" and not t.endswith("_x")]  # remove empty
                    vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding="utf-8") as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def write_sent_pairs(article, abstract, writer):
    # Write to tf.Example
    abstract = '%s %s %s' % (SENTENCE_START, abstract, SENTENCE_END)
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article.encode("utf-8")])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode("utf-8")])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))


def split_train_data(origin_data):
    with open(origin_data, encoding="utf-8") as fin, open(train_file,"w",encoding="utf-8") as f_train, open(val_file,"w",encoding="utf-8") as f_val,open(test_file,"w",encoding="utf-8") as f_test:
        index = 0;
        lines = fin.readlines()
        lines = [l for l in lines if len(l)>5 and not l.startswith("/x_S")]
        for count in range(TRAIN_SIZE):
            f_train.write(lines[index])
            index+=1
        for count in range(VAL_SIZE):
            f_val.write(lines[index])
            index += 1
        for count in range(TEST_SIZE):
            f_test.write(lines[index])
            index += 1


def make_vocab(vocab_file):
    vocab_counter = collections.Counter()

    with open(vocab_file, encoding="utf-8") as fin:
        for line in fin.readlines():
            tokens = line.strip().split(" ")
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != "" and not t.endswith("_x")]  # remove empty
            vocab_counter.update(tokens)

    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w', encoding="utf-8") as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
            writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")


if __name__ == '__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # cut("./datas/origin/data.train")
    make_vocab("./datas/origin/data.train.seg")
    # split_train_data(origin_data)

    # Read the text file, do a little postprocessing then write to bin files
    segment = True
    # write_to_bin(test_file, os.path.join(finished_files_dir, "test.bin"), segment=segment, repeat=1)
    # write_to_bin(val_file, os.path.join(finished_files_dir, "val.bin"), segment=segment, repeat=1)
    # write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), segment=segment, repeat=1)