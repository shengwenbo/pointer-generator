# -*-coding:utf-8-*-
import os
import struct
import collections
from tensorflow.core.example import example_pb2

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

origin_data = "./datas/origin/data.train"
train_file = "./datas/origin/train.txt"
val_file = "./datas/origin/val.txt"
test_file = "./datas/origin/test.txt"
finished_files_dir = "./datas/new"

TRAIN_SIZE = 700000
VAL_SIZE = 10000
TEST_SIZE = 1000

VOCAB_SIZE = 200000


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def write_to_bin(input_file, out_file, makevocab=False):
    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        # read the  input text file , make even line become article and odd line to be abstract（line number begin with 0）
        lines = read_text_file(input_file)

        count = 0
        for line in lines:
            [_, _, content] = line.strip().split("\t", 2);
            if content.find("\t") >= 0:
                [origin, corrects] = content.split("\t", 1)
                origin = " ".join(list(origin))
                if corrects:
                    corrects = corrects.split("\t")
                    for c in corrects:
                        c = " ".join(list(c))
                        write_sent_pairs(origin, c, writer)
                        # write_sent_pairs(c, c, writer)
                        count += 1
                        if count > 6000:
                            exit(1)
                else:
                    write_sent_pairs(origin, origin, writer)

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = origin.split(' ')
                    tokens = art_tokens
                    tokens = [t.strip() for t in tokens]  # strip
                    tokens = [t for t in tokens if t != ""]  # remove empty
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
        lines = fin.readlines();
        for count in range(TRAIN_SIZE):
            f_train.write(lines[index])
            index+=1
        for count in range(VAL_SIZE):
            f_val.write(lines[index])
            index += 1
        for count in range(TEST_SIZE):
            f_test.write(lines[index])
            index += 1


if __name__ == '__main__':

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    split_train_data(origin_data)

    # Read the text file, do a little postprocessing then write to bin files
    write_to_bin(test_file, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(val_file, os.path.join(finished_files_dir, "val.bin"))
    write_to_bin(train_file, os.path.join(finished_files_dir, "train.bin"), makevocab=True)