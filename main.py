import argparse

import sys
sys.path.insert(0, "./code/")

# from train_tuheeg_pathology_balanced_saturation_finalrun  import * 

from train_tuheeg_pathology_balanced_saturation_finalrun_mj  import * 


def main():
    # create a parser object
    parser = argparse.ArgumentParser(description="Train TUH EEG Pathology")

    # add arguments to the parser
    parser.add_argument( "--task_name", default="TUAB_finalrun_" ,type=str, help="task name")
    parser.add_argument( "--ids_to_load_train", default=None, type=int, help="ids to load train")
    parser.add_argument( "--model_name", default="TCN" ,type=str, help="model name")
    parser.add_argument( "--drop_prob", default=0.05270154233150525 ,type=float, help="drop prob")
    parser.add_argument( "--batch_size", default=64 ,type=int, help="batch size")
    parser.add_argument( "--lr", default=0.0011261049710243193 ,type=float, help="learning rate")
    parser.add_argument( "--n_epochs", default=35 ,type=int, help="number of epochs")
    parser.add_argument( "--weight_decay", default=5.83730537673086e-07 ,type=float, help="optimizer weight decay")
    parser.add_argument( "--seed", default=1000 ,type=int, help="random seed")
    parser.add_argument( "--cuda", default=True ,type=bool, help="use gpu or not")
    parser.add_argument( "--result_folder", default='~/scratch/medical/eeg/tuab/tuab_res/' ,type=str, help="result folder")
    parser.add_argument( "--train_folder", default='~/scratch/medical/eeg/tuab/tuab_pp2' ,type=str, help="data folder")
    parser.add_argument( "--train_folder2", default= None ,type=str, help="data folder")
    parser.add_argument( "--pre_trained", default=False ,type=bool, help="pre trained or not")
    parser.add_argument( "--load_path", default=None ,type=str, help="load path")

    # parse the arguments
    args = parser.parse_args()

    # print the arguments
    for k, v in vars (args).items ():
        print (f' {k} : {v}')

    if args.ids_to_load_train is not None:
        args.ids_to_load_train = range (args.ids_to_load_train)

    train_TUHEEG_pathology(
                        model_name= args.model_name,
                        task_name = args.task_name,
                        drop_prob= args.drop_prob,
                        batch_size= args.batch_size,
                        lr= args.lr,
                        n_epochs= args.n_epochs,
                        weight_decay=args.weight_decay,
                        result_folder= args.result_folder,
                        train_folder= args.train_folder,
                        # eval_folder= args.eval_folder,
                        ids_to_load_train =args.ids_to_load_train,
                        cuda = args.cuda,
                        seed= args.seed,
                        pre_trained = args.pre_trained,
                        load_path = args.load_path,
                        )

if __name__ == "__main__":
    main()











