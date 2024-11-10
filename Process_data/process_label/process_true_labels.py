'''
用于从比赛的test_label_A.npy中提取出真实标签, 并以
A<类序号>
的格式输出。
例如:
A009
A010
'''

import numpy as np;

# 对于0, 11这样不足三位的序号，获得用0填充后的序号: 000, 011
def get_full_length_label(label):
    fill_length = (3 - len(str(label)));
    zero = "0";
    result = zero * fill_length + str(label);
    return result;

if __name__ == "__main__":
    labels = np.load("test_label_A.npy");
    with open("true_labels.txt", "a") as file:
        for label in labels:
            file.write("A" + get_full_length_label(label) + "\n");