import torch
import pickle
import argparse
import numpy as np
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser(description = 'multi-stream ensemble') 
    parser.add_argument(
        '--ctrgcn_J3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument(
        '--ctrgcn_JBM3d_Score', 
        type = str,
        default = './Model_inference/Mix_GCN/output/uav_human/ctrgcn_V1_J_3D_bone_vel/epoch1_test_score.pkl')
    # Former, Joint,
    parser.add_argument(
        '--val_sample', 
        type = str,
        default = './Process_data/CS_test_V1.txt'),
    parser.add_argument(
        '--benchmark', 
        type = str,
        default = 'V1')
    return parser

def Cal_Score(File, Rate, ntu60XS_num, Numclass):
    final_score = torch.zeros(ntu60XS_num, Numclass)
    for idx, file in enumerate(File):
        fr = open(file,'rb') 
        inf = pickle.load(fr)

        df = pd.DataFrame(inf)
        df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        score = torch.tensor(data = df.values)
        final_score += Rate[idx] * score
    return final_score

def Cal_Acc(final_score, true_label):
    wrong_index = []
    _, predict_label = torch.max(final_score, 1)
    for index, p_label in enumerate(predict_label):
        if p_label != true_label[index]:
            wrong_index.append(index)
            
    wrong_num = np.array(wrong_index).shape[0]
    print('wrong_num: ', wrong_num)

    total_num = true_label.shape[0]
    print('total_num: ', total_num)
    Acc = (total_num - wrong_num) / total_num
    return Acc

def gen_label(val_txt_path):
    true_label = []
    val_txt = np.loadtxt(val_txt_path, dtype = str)
    for idx, name in enumerate(val_txt):
        label = int(name.split('A')[1][:3])
        true_label.append(label)
    
    true_label = torch.from_numpy(np.array(true_label))
    return true_label

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    # Mix_GCN Score File
    # 之后在这里拓展不同的分数
    j_file = args.ctrgcn_J3d_Score
    b_file = args.ctrgcn_JBM3d_Score
    
    val_txt_file = args.val_sample

    File = [j_file, b_file]    
    if args.benchmark == 'V1':
        Numclass = 155
        Sample_Num = 2000
        Rate = [0.6, 0.4]
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label(val_txt_file)
    
    if args.benchmark == 'V2':
        Numclass = 155
        Sample_Num = 6599
        Rate = [0.7, 0.7, 0.3, 0.3,
                0.3, 0.3, 0.3, 0.3,
                0.7, 0.7, 0.3, 0.3,
                0.05, 0.05, 0.05, 0.05] 
        final_score = Cal_Score(File, Rate, Sample_Num, Numclass)
        true_label = gen_label(val_txt_file)
    
    Acc = Cal_Acc(final_score, true_label)

    print('acc:', Acc)
