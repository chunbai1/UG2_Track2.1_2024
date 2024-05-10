import torch
import torch.nn as nn
import dataset
import argparse
from csv import reader

from CRNN.utils import strLabelConverter
from CRNN.crnn import CRNN

from ASTER.models.model_builder import ModelBuilder
from ASTER.utils import DataInfo, get_str_list

from DAN.utils import load_network, flatten_label


def arg_parse(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='/data/ug2+/data_dryrun_2.1/dryrun_release/', help = 'path to the images for evaluation')
    parser.add_argument('--label_file', default='/root/data/ug2+/final_data_2.1_2024/labels_2.1_final.csv', help = 'csv file containing the label info')
    parser.add_argument('--method', default = 'all', help = 'method for evaluation: [CRNN, DAN, ASTER, all]')
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--cpu', action = 'store_true')
    
    return parser.parse_args()


def evaluate_CRNN(args, result_file): 
    use_gpu = ~args.cpu
    
    print('Start evaluating with CRNN')
    transforms = dataset.transform_CRNN((100,32))
    textset = dataset.TextDataset(args.label_file,args.image_path,transform = transforms)
    data_loader = torch.utils.data.DataLoader(textset, batch_size=args.batch_size)
    
    converter = strLabelConverter(alphabet = '0123456789abcdefghijklmnopqrstuvwxyz')
    
    model = CRNN(32, 1, 37, 256)
    model.load_state_dict(torch.load('CRNN/crnn.pth'))
    
    if torch.cuda.is_available() and use_gpu: 
        model = model.cuda()
    
    model.eval()
    
    correct = 0
    if textset.phase == 'dry run': 
        log_file = open('log_CRNN.txt', 'w')
    
    for data in data_loader:      
        if textset.phase == 'dry run': 
            images, labels = data
        else: 
            images = data
        
        if torch.cuda.is_available() and use_gpu: 
            images = images.cuda()

        preds = model(images)
        preds_size = torch.IntTensor([preds.size(0)] * images.shape[0])

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        pred_strs = converter.decode(preds.data, preds_size.data, raw=False)
        
        for pred_str in pred_strs: 
            result_file.write(pred_str+'\n')
            
        if textset.phase == 'dry run':
            for pred_str, label in zip(pred_strs,labels): 
                if pred_str == label: 
                    correct += 1
                    log_file.write('label: {}, predicted: {}, correct! \n'.format(label,pred_str))
                else: 
                    log_file.write('label: {}, predicted: {}, incorrect! \n'.format(label,pred_str))
                    
    if textset.phase == 'dry run':                
        print('Finished evaluating with CRNN, result: {}/{}'.format(correct,textset.len))
        log_file.write('Result: {}/{}\n'.format(correct,textset.len))
        log_file.close()
    else: 
        print('Finished evaluating with CRNN')
    
    
def evaluate_DAN(args, result_file): 
    use_gpu = ~args.cpu
    
    print('Start evaluating with DAN')
    transforms = dataset.transform_DAN((128,32))
    textset = dataset.TextDataset(args.label_file,args.image_path,transform = transforms)
    data_loader = torch.utils.data.DataLoader(textset, batch_size=args.batch_size)
    
    models, encdec = load_network()
    for model in models: 
        model.eval()
        if torch.cuda.is_available() and use_gpu: 
            model = model.cuda()
    
    label = ['0123456789abcdefghijklmnopqrstuvwxyz']
    target = encdec.encode(label)
    label_flatten, length = flatten_label(target)
    length = length.unsqueeze(0)
    
    if torch.cuda.is_available() and use_gpu: 
        target = target.cuda()
        label_flatten = label_flatten.cuda()
    
    correct = 0
    if textset.phase == 'dry run': 
        log_file = open('log_DAN.txt', 'w')
    
    for data in data_loader:      
        if textset.phase == 'dry run': 
            images, labels = data
        else: 
            images = data
            
        if torch.cuda.is_available() and use_gpu: 
            images = images.cuda()

        features = models[0](images)
        A = models[1](features)
        output, out_length = models[2](features[-1], A, target, length, True)

        pred_strs = encdec.decode(output,out_length)[0]

        for pred_str in pred_strs: 
            result_file.write(pred_str+'\n')
            
        if textset.phase == 'dry run':
            for pred_str, label in zip(pred_strs,labels): 
                if pred_str == label: 
                    correct += 1
                    log_file.write('label: {}, predicted: {}, correct! \n'.format(label,pred_str))
                else: 
                    log_file.write('label: {}, predicted: {}, incorrect! \n'.format(label,pred_str))
                    
    if textset.phase == 'dry run':                
        print('Finished evaluating with DAN, result: {}/{}'.format(correct,textset.len))
        log_file.write('Result: {}/{}\n'.format(correct,textset.len))
        log_file.close()
    else: 
        print('Finished evaluating with DAN')
    
    
def evaluate_ASTER(args, result_file): 
    use_gpu = ~args.cpu
    
    if torch.cuda.is_available() and use_gpu: 
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
    print('Start evaluating with ASTER')
    transforms = dataset.transform_ASTER((100,32))
    textset = dataset.TextDataset(args.label_file,args.image_path,transform = transforms)
    data_loader = torch.utils.data.DataLoader(textset, batch_size=args.batch_size)
    
    dataset_info = DataInfo('ALLCASES_SYMBOLS')
    model = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=dataset_info.rec_num_classes,
                           sDim=512, attDim=512, max_len_labels=100,
                           eos=dataset_info.char2id[dataset_info.EOS], STN_ON=True)
    model.load_state_dict(torch.load('ASTER/demo.pth.tar')['state_dict'])
    model.eval()

    if torch.cuda.is_available() and use_gpu: 
        model = model.cuda()
                  
    correct = 0
    if textset.phase == 'dry run': 
        log_file = open('log_ASTER.txt', 'w')
    
    for data in data_loader: 
        if textset.phase == 'dry run': 
            images, labels = data
        else: 
            images = data
        
        if torch.cuda.is_available() and use_gpu: 
            images = images.cuda()
        
        input_dict = {}
        input_dict['images'] = images
        rec_targets = torch.IntTensor(images.shape[0], 100).fill_(1)
        rec_targets[:,100-1] = dataset_info.char2id[dataset_info.EOS]
        rec_targets = rec_targets.cuda()
        input_dict['rec_targets'] = rec_targets
        input_dict['rec_lengths'] = [100]

        output_dict = model(input_dict)
        
        pred_strs, _ = get_str_list(output_dict['output']['pred_rec'], input_dict['rec_targets'], dataset=dataset_info)
        
        for pred_str in pred_strs: 
            result_file.write(pred_str+'\n')
            
        if textset.phase == 'dry run':
            for pred_str, label in zip(pred_strs,labels): 
                if pred_str == label: 
                    correct += 1
                    log_file.write('label: {}, predicted: {}, correct! \n'.format(label,pred_str))
                else: 
                    log_file.write('label: {}, predicted: {}, incorrect! \n'.format(label,pred_str))
                    
    if textset.phase == 'dry run':                
        print('Finished evaluating with ASTER, result: {}/{}'.format(correct,textset.len))
        log_file.write('Result: {}/{}\n'.format(correct,textset.len))
        log_file.close()
    else: 
        print('Finished evaluating with ASTER')
    

if __name__ == '__main__': 
    args = arg_parse()
    
    result_file = open('./result.txt','w')
    
    if args.method == 'CRNN': 
        evaluate_CRNN(args, result_file)
    elif args.method == 'DAN': 
        evaluate_DAN(args, result_file)
    elif args.method == 'ASTER': 
        evaluate_ASTER(args, result_file)
    elif args.method == 'all': 
        evaluate_CRNN(args, result_file)
        evaluate_DAN(args, result_file)
        evaluate_ASTER(args, result_file)
        
    result_file.close()
