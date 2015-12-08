#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os

def scientificNotation(value):
    if value == 0:
        return '0'
    else:
        e = np.log10(np.abs(value))
        m = np.sign(value) * 10 ** (e - int(e))
        return r'${:.0f} \cdot 10^{{{:d}}}$'.format(m, int(e))

def main():

    # import data
    parser = argparse.ArgumentParser(description='Process Input')
    parser.add_argument('-d', '--dir', type=str, required=True, help='Directory Of Probability CSV files')
    parser.add_argument('-v', '--verbose',action='store_true', help='Verbose output that shows each graph before saving')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print "error Firectory doesn't exist"
        return -1
    if os.path.isfile(args.dir):
        print "Error: Expected Directory not File"
        return -1
    for dirname, subdirlist, fileList, in os.walk(args.dir):
        if 'probability_report.csv' in fileList:
            filename = os.path.join(dirname, 'probability_report.csv')
            print "Processing: %s"%(filename)
            # graph predictions
            data = np.genfromtxt(filename, delimiter=',')[1:]
            block_data = data[data[:,4] == 5]
            ram_data   = data[data[:,4] == 1]
            herd_data  = data[data[:,4] == 7]
            '''
            plot a 4 sublpot graph with one combined graph and 3 of the individual
            probabilities to understand meaning easier
            '''
            fig = plt.figure(1,figsize=(16, 9), dpi=72)

            file = os.path.basename(os.path.dirname(filename))
            fig.suptitle("%s"%(file), fontsize=22)
            formatter = mpl.ticker.FuncFormatter(lambda x, p: scientificNotation(x))
            fig.text(0.5, 0.03, 'Time', ha='center', va='center', fontsize=14, fontweight='bold')
            fig.text(0.03, 0.5, 'Probability', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
            plt.subplots_adjust(hspace=.2, left=.06, top=.90, right=0.97, bottom=0.07)

            g1 = plt.subplot(411)
            plt.plot(block_data[:,0], block_data[:,7], label='Block', color='blue', linewidth=2.0)
            plt.plot(ram_data[:,0], ram_data[:,7], label='Ram', color='red', linewidth=2.0)
            plt.plot(herd_data[:,0], herd_data[:,7], label='Herd', color='green', linewidth=2.0)
            plt.title('Combined Prediction')
            plt.ylim([0,1])
            plt.legend(loc='upper right')
            plt.setp(g1.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_major_formatter(formatter)

            g2 = plt.subplot(412)
            plt.plot(block_data[:,0], block_data[:,7], color='blue', linewidth=2.0)
            plt.title('Block Prediction')
            plt.ylim([0,1])
            plt.setp(g2.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_major_formatter(formatter)

            g3 = plt.subplot(413)
            plt.plot(ram_data[:,0], ram_data[:,7], color='red', linewidth=2.0)
            plt.title('Ram Prediction')
            plt.ylim([0,1])
            plt.setp(g3.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_major_formatter(formatter)

            g4 = plt.subplot(414)
            plt.plot(herd_data[:,0], herd_data[:,7], color='green', linewidth=2.0)
            plt.title('Herd Prediction')
            plt.ylim([0,1])
            plt.gca().xaxis.set_major_formatter(formatter)

            # plt.tight_layout()

            plt.savefig(file + '.pdf')
            if args.verbose:
                plt.show()
            fig.clear()

if __name__ == '__main__':
    main()