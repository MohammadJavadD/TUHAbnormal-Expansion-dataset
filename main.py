import argparse

import sys
sys.path.insert(0, "./code/")

# from train_tuheeg_pathology_balanced_saturation_finalrun  import * 

from train_tuheeg_pathology_balanced_saturation_finalrun_mj  import train_TUHEEG_pathology 

from misc import defualt_parser

def main():

    # Create a parser object
    parser = argparse.ArgumentParser(description="Train TUH EEG Pathology")

    # Project and task related arguments
    parser.add_argument("--project_name", default="APD_revised_CpLoss", type=str, help="project name")
    parser.add_argument("--task_name", default="TUAB_finalrun_", type=str, help="task name")
    parser.add_argument("--target_name", default="pathological", type=str, help="classification target")
    parser.add_argument("--only_eval", default=False, type=bool, help="only eval or not")

    # Training data related arguments
    parser.add_argument("--ids_to_load_train", default=None, type=int, help="ids to load train, 1-2717")
    parser.add_argument("--ids_to_load_train2", default=None, type=int, help="ids to load train, 1-2171")
    parser.add_argument("--ids_to_load_train3", default=None, type=int, help="ids to load train, 1-2717")
    parser.add_argument("--ids_to_load_train4", default=None, type=int, help="ids to load train, 1-2171")
    parser.add_argument("--train_folder", default='~/scratch/medical/eeg/tuab/tuab_pp2', type=str, help="data folder")
    parser.add_argument("--train_folder2", default=None, type=str, help="data folder")
    parser.add_argument("--train_folder3", default=None, type=str, help="data folder")
    parser.add_argument("--train_folder4", default=None, type=str, help="data folder")

    # Model related arguments
    parser.add_argument("--model_name", default="TCN", type=str, help="model name")
    parser.add_argument("--n_tcn_blocks", default=4, type=int, help="number of TCN blocks")
    parser.add_argument("--n_tcn_filters", default=55, type=int, help="number of TCN filters")
    parser.add_argument("--drop_prob", default=0.05270154233150525, type=float, help="dropout probability")

    # Training process related arguments
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0011261049710243193, type=float, help="learning rate")
    parser.add_argument("--n_epochs", default=35, type=int, help="number of epochs")
    parser.add_argument("--weight_decay", default=5.83730537673086e-07, type=float, help="optimizer weight decay")
    parser.add_argument("--seed", default=1000, type=int, help="random seed")
    parser.add_argument("--cuda", default=True, type=bool, help="use gpu or not")
    parser.add_argument("--pre_trained", default=False, type=bool, help="use pre-trained model or not")
    parser.add_argument("--augment", default=False, type=bool, help="use data augmentation or not")

    # Output related arguments
    parser.add_argument("--result_folder", default='~/scratch/medical/eeg/tuab/tuab_res/', type=str, help="result folder")
    parser.add_argument("--load_path", default=None, type=str, help="path to load pre-trained model")

    # Misc
    parser.add_argument("--use_defualt_parser", default=False, type=bool, help="use default parser or not")

    # Parse the arguments
    args = parser.parse_args()


    if args.use_defualt_parser:
        print("use defualt parser")
        args = defualt_parser(args)

    ## add wandb ##
    # Install wandb
    # pip install wandb
    import wandb

    # Create a wandb Run
    wandb_run = wandb.init(
        # Set the project where this run will be logged
        project=args.project_name,
        name=args.task_name,
        # Track hyperparameters and run metadata
        config= vars(args),
    )
    ## end of add wandb ##

    # print the arguments
    for k, v in vars (args).items ():
        print (f' {k} : {v}')

    b_acc_merge, b_acc_tuh, b_acc_nmt, loss_merge, loss_tuh, loss_nmt = train_TUHEEG_pathology(
                        model_name= args.model_name,
                        task_name = args.task_name,
                        target_name= args.target_name,
                        drop_prob= args.drop_prob,
                        batch_size= args.batch_size,
                        lr= args.lr,
                        n_epochs= args.n_epochs,
                        weight_decay=args.weight_decay,
                        result_folder= args.result_folder,
                        train_folder= args.train_folder,
                        train_folder2= args.train_folder2,
                        train_folder3= args.train_folder3,
                        train_folder4= args.train_folder4,
                        # eval_folder= args.eval_folder,
                        ids_to_load_train =args.ids_to_load_train,
                        ids_to_load_train2 =args.ids_to_load_train2,
                        ids_to_load_train3 =args.ids_to_load_train3,
                        ids_to_load_train4 =args.ids_to_load_train4,
                        cuda = args.cuda,
                        seed= args.seed,
                        pre_trained = args.pre_trained,
                        load_path = args.load_path,
                        augment = args.augment,
                        n_tcn_blocks = args.n_tcn_blocks,
                        n_tcn_filters = args.n_tcn_filters,
                        wandb_run = wandb_run,
                        only_eval = args.only_eval
                        )
    wandb.run.summary["loss_merge"] = loss_merge
    wandb.run.summary["loss_tuh"] = loss_tuh
    wandb.run.summary["loss_nmt"] = loss_nmt
    wandb.run.summary["b_acc_merge"] = b_acc_merge
    wandb.run.summary["b_acc_tuh"] = b_acc_tuh
    wandb.run.summary["b_acc_nmt"] = b_acc_nmt
    
if __name__ == "__main__":
    main()











