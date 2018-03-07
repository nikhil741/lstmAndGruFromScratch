import matplotlib.pyplot as plt
import os

if not os.path.exists("../output"):
	os.mkdir("../output")

fp = open("./hiddenLayer_Accuracy_Values.txt")

allLines = fp.readlines()

lstm_x = []
lstm_y = []
gru_x = []
gru_y = []

for line in allLines:
	#print line
	detailList = line.split("\t")
	if detailList[0] == "lstm":
		lstm_x.append(float(detailList[1]))
		lstm_y.append(float(detailList[2].strip('\n')))
	if detailList[0] == "gru":
		gru_x.append(float(detailList[1]))
		gru_y.append(float(detailList[2].strip("\n")))


print lstm_x
print lstm_y
print gru_x
print gru_y

# giving a title to my graph
plt.title('Weights vs Accuracy in LSTM & GRU')

plt.plot(lstm_x, lstm_y, label="LSTM")
plt.plot(gru_x, gru_y, label="GRU")

# naming the x axis
plt.xlabel('Weights')

# naming the y axis
plt.ylabel('Accuracy')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.savefig("../output/graphAccuracyVsHiddenUnits")