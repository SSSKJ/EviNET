import torch
import numpy as np



for dataset in ["amazon-photo", "amazon-computer", "coauthor-cs", "coauthor-physics"]: # "cora", "citeseer", "amazon-photo", "amazon-computer", "coauthor-cs", "coauthor-physics", "arxiv"

    datasetname = dataset

    best_acc = []
    best_aurc = []
    best_fpr = []
    best_auroc = []
    best = -10000

    best_file = ''

    if dataset in ["cora", "amazon-photo", "coauthor-cs"]:
        datasetname += '-label'

    for noise in ['0.0001', '0.01', '0.1', '0.5', '1']:
        for T in [1, 5, 10, 20, 50, 100]:

            filename = f'./results/{datasetname}/ODIN_all.pth'
            # filename = f'./results/{datasetname}/ODIN_{noise}_{T}.pth'


            acc = []
            aurc = []
            fpr = []
            auroc = []
            for r in torch.load(filename):
                
                acc.append(r['Acc'])
                aurc.append(r['AURC'])
                fpr.append(r['FPR95'])
                auroc.append(r['AUROC'])

                if np.array(acc).mean() + np.array(auroc).mean() - np.array(aurc).mean() > best:
                    best = np.array(acc).mean() + np.array(auroc).mean() - np.array(aurc).mean()
                    best_acc = acc
                    best_aurc = aurc
                    best_fpr = fpr
                    best_auroc = auroc
                    best_file = filename
            
            # print(f'------------------{dataset}-------------------')
            # print(f'Acc: {np.array(acc).mean():.2f} ± {np.array(acc).std():.2f}')
            # print(f'AURC: {np.array(aurc).mean():.2f} ± {np.array(aurc).std():.2f}')
            # print(f'fpr: {np.array(fpr).mean():.2f} ± {np.array(fpr).std():.2f}')
            # print(f'auroc: {np.array(auroc).mean():.2f} ± {np.array(auroc).std():.2f}')
        
    print(f'------------------{dataset}-{best_file}-------------------')
    print(f'Acc: {np.array(best_acc).mean():.2f} ± {np.array(best_acc).std():.2f}')
    print(f'AURC: {np.array(best_aurc).mean():.2f} ± {np.array(best_aurc).std():.2f}')
    print(f'fpr: {np.array(best_fpr).mean():.2f} ± {np.array(best_fpr).std():.2f}')
    print(f'auroc: {np.array(best_auroc).mean():.2f} ± {np.array(best_auroc).std():.2f}')
