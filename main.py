import time
from utils import data_loader, args_parser, target_detectors
import numpy as np
import spectral.io.envi as envi
import os
import matplotlib.pyplot as plt
from sklearn import metrics

args = args_parser.args_parser()
print (args)

def main():
    img, tgt_sig, tgt_mask = data_loader.load_dataset(args.hsidataset, args.targetsig, args.root)
    
    #Calculate matched filter scores
    mf_scores  = target_detectors.calc_mf(img, tgt_sig)
    
    #Calculate ace_scores
    ace_scores = target_detectors.calc_ace(img, tgt_sig)
    
    #Calculate sam_scores
    sam_scores = target_detectors.calc_sam(img, tgt_sig)
    
    #Visualize target detection using ROC Curves
    plt.figure()
    plt.title('Data: ' + args.hsidataset + '; Target: ' + args.targetsig)
    y_true       = tgt_mask.ravel()
    y_scores_mf  = mf_scores.ravel()
    y_scores_ace = ace_scores.ravel()
    y_scores_sam = sam_scores.ravel()
    
    #Use sklearn 'metrics'
    fpr_mf,  tpr_mf,  _ = metrics.roc_curve(y_true, y_scores_mf)
    fpr_ace, tpr_ace, _ = metrics.roc_curve(y_true, y_scores_ace)
    fpr_sam, tpr_sam, _ = metrics.roc_curve(y_true, y_scores_sam)
    
    plt.plot(fpr_mf, tpr_mf,   label='mf',  c='b')
    plt.plot(fpr_ace, tpr_ace, label='ace', c='g')
    plt.plot(fpr_sam, tpr_sam, label='sam', c='r')
    
    plt.xlabel('False alarm rate')
    plt.ylabel('Detection rate')
    plt.xscale('log')
    
    plt.legend()
    #plt.show()
    
    #Save plot
    fig_pth = './figures/' + args.hsidataset + '_' + args.targetsig + 'felt.png'
    plt.savefig(fig_pth)

if __name__ == '__main__':
    main()
