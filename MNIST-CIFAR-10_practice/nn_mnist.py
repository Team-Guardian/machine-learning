from utilities import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

# Load MNIST datasets and apply min/max scaling
# Pixel intensity [0, 1]
# 8 * 8 (64 dimensions)

print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")

# This normalizes the data
data = (data - data.min()) / (data.max() - data.min())

# Construct training/ test split.
(trainX, testX, trainY, testY) = train_test_split(
    data, digits.target, test_size = 0.25)

# Binarize our label.
# One-hot encoding
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] training network...")
#trainX.shape[1] is 64, (64 nodes in input layer)
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=200)

print("[INFO] evaluating network...")
predictions = nn.predict(testX)
# This picks out the largest class
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
