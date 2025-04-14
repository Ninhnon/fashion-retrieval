import os
from pathlib import Path
from argparse import ArgumentParser

# Default paths relative to project root
DEFAULT_GROUNDTRUTH_ROOT = Path('dataset/groundtruth')
DEFAULT_EVALUATION_ROOT = Path('dataset/evaluation')

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

def compute_mAP(feature_extractor, groundtruth_root, evaluation_root, crop=False):
    # Construct evaluation path
    path_evaluation = Path(evaluation_root)
    path_evaluation = path_evaluation / ('crop' if crop else 'original')
    path_evaluation = path_evaluation / feature_extractor

    AP = 0.0
    number_query = 0.0

    for query in os.listdir(str(path_evaluation)):
        groundtruth_path = Path(groundtruth_root)
        good_path = groundtruth_path / f"{query[:-4]}_good.txt"
        ok_path = groundtruth_path / f"{query[:-4]}_ok.txt"

        with open(good_path, 'r') as file:
            good_set = file.read().split('\n')
        with open(ok_path, 'r') as file:
            ok_set = file.read().split('\n')
            
        # positive set of ground truth = ok_set + good_set
        pos_set = ok_set + good_set

        with open(path_evaluation / query) as file:
            ranked_list = file.read().split('\n')
        
        AP += compute_AP(pos_set, ranked_list)
        number_query += 1
    
    return AP / number_query

def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", type=str, default='Resnet50',
                      help="Feature extractor to use: Resnet50, EfficientNetV2, VIT, RGBHistogram, or LBP")
    parser.add_argument("--crop", type=bool, default=False,
                      help="Whether to use cropped images for evaluation")
    parser.add_argument("--groundtruth_root", type=str, default=str(DEFAULT_GROUNDTRUTH_ROOT),
                      help="Path to directory containing groundtruth files")
    parser.add_argument("--evaluation_root", type=str, default=str(DEFAULT_EVALUATION_ROOT),
                      help="Path to directory containing evaluation results")

    args = parser.parse_args()
    
    AP = compute_mAP(
        args.feature_extractor,
        args.groundtruth_root,
        args.evaluation_root,
        args.crop
    )
    
    print(f'Mean Average Precision (mAP) for {args.feature_extractor}: {AP:.4f}')

if __name__ == '__main__':
    main()
