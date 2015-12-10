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
        print "error directory doesn't exist"
        return -1
    if os.path.isfile(args.dir):
        print "Error: Expected Directory not File"
        return -1

    accuracy_list = []
    early_predict_list = []
    early_predict_list_2 = []
    changes_list = []
    filename_list = []
    for dirname, subdirlist, fileList, in os.walk(args.dir):
        if 'probability_report.csv' in fileList:
            filename = os.path.join(dirname, 'probability_report.csv')
            print "Processing: %s"%(filename)
            # graph predictions
            data = np.genfromtxt(filename, delimiter=',')[1:]
            block_data = data[data[:,6] == 5]
            ram_data   = data[data[:,6] == 1]
            herd_data  = data[data[:,6] == 7]
            '''
            plot a 4 sublpot graph with one combined graph and 3 of the individual
            probabilities to understand meaning easier
            '''
            fig = plt.figure(1,figsize=(16, 9), dpi=72)

            file = os.path.basename(os.path.dirname(filename))
            filename_list.append(file)
            fig.suptitle("%s"%(file), fontsize=22)
            formatter = mpl.ticker.FuncFormatter(lambda x, p: scientificNotation(x))
            fig.text(0.5, 0.03, 'Time', ha='center', va='center', fontsize=14, fontweight='bold')
            fig.text(0.03, 0.5, 'Probability', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
            plt.subplots_adjust(hspace=.2, left=.06, top=.90, right=0.97, bottom=0.07)

            # g1 = plt.subplot(411)
            # plt.plot(block_data[:,0], block_data[:,7], label='Block', color='blue', linewidth=2.0)
            # plt.plot(ram_data[:,0], ram_data[:,7], label='Ram', color='red', linewidth=2.0)
            # plt.plot(herd_data[:,0], herd_data[:,7], label='Herd', color='green', linewidth=2.0)
            # plt.title('Overlayed Predictions')
            # plt.ylim([-0.01,1.01])
            # plt.legend(loc='upper right')
            # plt.setp(g1.get_xticklabels(), visible=False)
            # plt.gca().xaxis.set_major_formatter(formatter)

            g2 = plt.subplot(311)
            plt.plot(block_data[:,0], block_data[:,7], color='blue', linewidth=2.0)
            plt.title('Block Prediction')
            plt.ylim([-0.01,1.01])
            plt.setp(g2.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_major_formatter(formatter)

            g3 = plt.subplot(312)
            plt.plot(ram_data[:,0], ram_data[:,7], color='red', linewidth=2.0)
            plt.title('Ram Prediction')
            plt.ylim([-0.01,1.01])
            plt.setp(g3.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_major_formatter(formatter)

            g4 = plt.subplot(313)
            plt.plot(herd_data[:,0], herd_data[:,7], color='green', linewidth=2.0)
            plt.title('Herd Prediction')
            plt.ylim([-0.01,1.01])
            plt.gca().xaxis.set_major_formatter(formatter)

            # plt.tight_layout()

            plt.savefig(file + '.pdf')
            if args.verbose:
                plt.show()
            fig.clear()
            # Generate Timeline predictions

            # Determine Correct prediction
            if 'block' in file.lower():
                # print "Block Type"
                class_type = 5
            elif 'ram' in file.lower():
                # print "Ram Type"
                class_type = 1
            elif 'herd' in file.lower():
                # print "Herd Type"
                class_type = 7
            else:
                print "Error: Type not detected"
            prediction = []

            for i in xrange(len(block_data)):
                m = max(block_data[i, 7], ram_data[i, 7], herd_data[i,7])
                if m == block_data[i, 7]:
                    prediction.append(block_data[i])
                elif m == ram_data[i, 7]:
                    prediction.append(ram_data[i])
                elif m == herd_data[i, 7]:
                    prediction.append(herd_data[i])
            prediction = np.array(prediction)
            # Compute Accuracy
            # block_num = len([prediction[i] for i in xrange(len(prediction)) if prediction[i,4] == 5])
            # ram_num = len([prediction[i] for i in xrange(len(prediction)) if prediction[i,4] == 1])
            # herd_num = len([prediction[i] for i in xrange(len(prediction)) if prediction[i,4] == 7])
            num_correct = len([prediction[i] for i in xrange(len(prediction)) if prediction[i,6] == class_type])
            num_predict = len(prediction)
            accuracy = float(num_correct) / float(num_predict)
            accuracy_list.append(accuracy)

            # Compute confusion matrix
            ground_truth = [class_type for i in xrange(len(prediction))]
            class_types = [1,5,7]

            confusion_mat = np.zeros([3,3])
            for i, correct_index in enumerate(class_types):
                for j, predicted_index in enumerate(class_types):
                    for k in xrange(len(prediction)):
                        if prediction[k,6] == predicted_index and ground_truth[k] == correct_index:
                            confusion_mat[i, j] += 1.0
            for i, row in enumerate(confusion_mat):
                confusion_mat[i] = row / sum(row, 1)

            fig_conf = plt.figure(2)

            ax = fig_conf.add_subplot(111)
            ax.set_aspect(1)
            rows = len(confusion_mat)
            cols = len(confusion_mat[0])
            res = ax.imshow(confusion_mat, cmap=plt.cm.jet, interpolation='nearest')

            for x in xrange(cols):
                for y in xrange(rows):
                    ax.annotate(str(confusion_mat[y,x]), xy=(x,y),horizontalalignment='center',verticalalignment='center')

            cb = fig_conf.colorbar(res)
            class_list = ['RAM', 'BLOCK', 'HERD']
            plt.xticks(range(len(class_list)), class_list)
            plt.yticks(range(len(class_list)), class_list)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.title('%s Confusion Matrix'%(file))
            plt.savefig(file + '_confusion.pdf')
            if args.verbose:
                plt.show()
            fig_conf.clear()

            # Compute Early Detection Rate
            early_predict = 1
            for i in reversed(xrange(len(prediction))):
                if prediction[i,6] != class_type:
                    early_predict = float((i+1)) / float(num_predict)
                    break
            early_predict_list.append(early_predict)
            # Compute Second early detection Rate
            early_predict_2 = 1
            for i in xrange(len(prediction)):
                if prediction[i,6] == class_type:
                    early_predict_2 = float(i) / float(num_predict)
                    break
            early_predict_list_2.append(early_predict_2)
            # compute Persistance of Recognized intent
            changes = -1
            previous_state = -1
            for row in prediction:
                if previous_state != row[6]:
                    changes += 1
                previous_state = row[6]
            changes_list.append(changes)

            with open(file + '.csv', 'w') as openfile:
                openfile.write('Accuracy, Early Detection, First Detection, Persistence\n')
                openfile.write('%f, %f, %f, %f\n'%(accuracy, early_predict, early_predict_2, changes))

    # Compute Accuracy mean/std
    accuracy_array = np.array(accuracy_list)
    acc_mean = np.mean(accuracy_array)
    acc_std = np.std(accuracy_array)

    # Compute Early Detection mean/std
    early_predict_array = np.array(early_predict_list)
    early_mean = np.mean(early_predict_array)
    early_std = np.std(early_predict_array)

    # Compute Early first detection
    early_predict_array_2 = np.array(early_predict_list_2)
    early_mean_2 = np.mean(early_predict_array_2)
    early_std_2 = np.std(early_predict_array_2)

    # Compute Persistence mean/std
    changes_array = np.array(changes_list)
    change_mean = np.mean(changes_array)
    change_std = np.std(changes_array)

    with open('all_data_analysis.csv', 'w') as openfile:
        openfile.write('Type, Mean, Standard Deviation\n')
        openfile.write('Accuracy, %f, %f\n'%(acc_mean, acc_std))
        openfile.write('Early Detection, %f, %f\n'%(early_mean, early_std))
        openfile.write('First Detection, %f, %f\n'%(early_mean_2, early_std_2))
        openfile.write('Persistence, %f, %f\n'%(change_mean, change_std))
        openfile.write('\n')

        openfile.write('LOG, Accuracy, Early Detection, First Detection, Persistence\n')
        for i in xrange(len(filename_list)):
            openfile.write('%s, %f, %f, %f, %f\n'%(filename_list[i], accuracy_list[i], early_predict_list[i], early_predict_list_2[i], changes_list[i]))


if __name__ == '__main__':
    main()