from argparse import ArgumentParser
import os

root_groundtruth = 'D:/DoAn/DoAn2/Paper/image-retrieval-main/image-retrieval-main/dataset/groundtruth'
root_evaluation = 'D:/DoAn/DoAn2/Paper/image-retrieval-main/image-retrieval-main/dataset/evaluation'

def compute_AP(pos_set, ranked_list):
    relevant = 0.0  # Changed from 1.0 to 0.0
    average_precision = 0.0
    number_retrieve = 0

    for item in ranked_list:
        number_retrieve += 1
        if item not in pos_set:
            continue
        
        relevant += 1
        average_precision += (relevant/number_retrieve)
    
    # Avoid division by zero
    if relevant == 0:
        return 0.0
        
    return average_precision / relevant

def compute_mAP(feature_extractor, crop = False):
    if (crop):
        path_evaluation =  root_evaluation + '/crop'
    else:
        path_evaluation = root_evaluation + '/original'

    path_evaluation += ('/' + feature_extractor)

    AP = 0.0
    number_query = 0.0

    for query in os.listdir(path_evaluation):
        with open(root_groundtruth + '/' + query[:-4] + '_good.txt', 'r') as file:
            good_set = file.read().split('\n')
        with open(root_groundtruth + '/' + query[:-4] + '_ok.txt', 'r') as file:
            ok_set = file.read().split('\n')
            
        # positive set of ground truth = ok_set + good_set
        pos_set = ok_set + good_set

        with open(path_evaluation + '/' + query) as file:
            ranked_list = file.read().split('\n')
        
        AP += compute_AP(pos_set, ranked_list)
        number_query += 1
    
    return AP / number_query

def main():
        parser = ArgumentParser()
        parser.add_argument("--feature_extractor", required=False, type=str, default='Resnet50')
        parser.add_argument("--crop", required=False, type=bool, default=False)

        args = parser.parse_args()
        AP = compute_mAP(args.feature_extractor, args.crop)
        print(f'mAP of {args.feature_extractor} is {AP}')

if __name__ == '__main__':
    main()