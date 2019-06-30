using namespace std;

/** 
 * @title 
 *      Neural Network Model
 * @description 
 *      Parts of this code are inspired from the Tremani Neural Network.
 * @author 
 *      Chester J. Ayala Jr.
 */

class NeuralNetwork {
    private:
        vector<int> nodeCount;
        vector<vector<double>> nodeValue;
        vector<vector<double>> controlData;
        vector<vector<vector<double>>> nodeWeight;
        vector<vector<double>> nodeThreshold;
        vector<vector<double>> nodeWeightCorrection;
        double errorTrainingSet;
        double errorControlSet;
        double learningRate;
        double momentum;
        int epoch;
        void initializeWeights();
        void initializeNodes();
        vector<double> feedforward(vector<double> input);
        void backpropagate(vector<double> output, vector<double> targetOutput);
        double activation(double nodeValue);
        double derivativeActivation(double value);
        double generateRandomWeight();
        double squaredErrorEpoch(vector<vector<double>> trainingData);
        double squaredError(vector<double> input, vector<double> targetOutput);
        double squaredErrorControlSet();
        void fitLine(vector<double> avgErrorControlSet, double *slope, double *offset);
        bool success = false;

    public:
        NeuralNetwork(int inputCount, vector<int> hiddenCount, int outputCount);
        int getNodeCount(int layer);
        bool train(vector<vector<double>> trainingData = {{0}}, int maxEpoch = 500, double maxRMSE = 0.10, double learningRate = 0.1, double momentum = 0.8);
};

/**
 * Initialize layers
 * 
 * @param int inputCount
 *      The number of nodes in the input layer
 * @param int hiddenCount[layer]
 *      The number of nodes in the hidden layer
 * @param int outputCount
 *      The number of nodes in the output layer
 */
NeuralNetwork::NeuralNetwork(int inputCount, vector<int> hiddenCount, int outputCount) {
    this->nodeCount.push_back(inputCount);

    if (hiddenCount[0] == 0) {
        // Reserved for automatic computation of number of hidden layers and nodes based on input/output node size
    } 
    else {
        for(int layer = 0; layer < hiddenCount.size(); layer++) {
            this->nodeCount.push_back(hiddenCount[layer]);
        }
    }
    this->nodeCount.push_back(outputCount);
    this->initializeNodes();
    this->initializeWeights();
}

/**
 * Initialize the nodes with zero values
 */
void NeuralNetwork::initializeNodes() {
    vector<vector<double>> nodeValue;
    for (int layer = 0; layer < this->nodeCount.size(); layer++) {
        vector<double> zeroValue(this->nodeCount[layer]);
        nodeValue.push_back(zeroValue);
    }
    this->nodeValue = nodeValue;
    this->nodeWeightCorrection = nodeValue;
}

/**
 * Initialize the weights with random values
 */
void NeuralNetwork::initializeWeights() {
    vector<vector<vector<double>>> nodeWeight;
    vector<vector<double>> nodeThreshold;
    for (int layer = 0; layer < this->nodeCount.size()+1; layer++) {
        vector<vector<double>> sourceNodeWeight;
        vector<double> threshold;
        for (int sourceNode = 0; sourceNode < this->nodeCount[layer]; sourceNode++) {
            vector<double> destinationNodeWeight;
            for (int destinationNode = 0; destinationNode < this->nodeCount[layer+1]; destinationNode++) {
                destinationNodeWeight.push_back(this->generateRandomWeight());
            }
            sourceNodeWeight.push_back(destinationNodeWeight);
            threshold.push_back(this->generateRandomWeight());
        }
        nodeWeight.push_back(sourceNodeWeight);
        nodeThreshold.push_back(threshold);
    }

    this->nodeWeight = nodeWeight;
    this->nodeThreshold = nodeThreshold;
}

/**
 * Get the node count of the layer
 * 
 * @param int layer
 *      The layer to get the node count from
 */
int NeuralNetwork::getNodeCount(int layer) {
    return this->nodeCount[layer];
}

/**
 * Train the model with training data
 * 
 * @param double trainingData[row][column]
 *      The data that will be used to train the model
 * @param int maxEpoch
 *      ...
 * @param double maxRMSE
 *      ...
 * @param double learningRate
 *      ...
 * @param double momentum
 *      ...
 * @return bool
 *      Determines whether the training is successful
 */
bool NeuralNetwork::train(vector<vector<double>> trainingData, int maxEpoch, double maxRMSE, double learningRate, double momentum) {
    if (trainingData[0].size() != this->nodeCount[0] + this->nodeCount[this->nodeCount.size()-1]) {
        cout<<"Data does not match the size for the Input and Output of the Model."<<endl;
        return false;
    }
    this->learningRate = learningRate;
    this->momentum = momentum;
    vector<double> errorControlSet;
    vector<double> avgErrorControlSet;
    double slope = 0.0;
    double offset = 0.0;
    double squaredError = 0.0;
    double squaredErrorControlSet = 0.0;
    int epoch = 0;
    int sampleCount = 10;
    bool condition1 = false, condition2 = false, condition3 = false;
    do {
        for (int iteration = 0; iteration < trainingData.size(); iteration++) {
            int index = rand() % (trainingData.size()-1 + 1);
            vector<double> data = trainingData[index];
            vector<double> input;
            vector<double> targetOutput;
            for (int index = 0; index < this->nodeCount[0]; index++) {
                input.push_back(data[index]);
            }
            for (int index = this->nodeCount[0]; index < this->nodeCount[0] + this->nodeCount[this->nodeCount.size()-1]; index++) {
                targetOutput.push_back(data[index]);
            }
            vector<double> output = feedforward(input);
            this->backpropagate(output, targetOutput);
        }
        squaredError = this->squaredErrorEpoch(trainingData);
        if (epoch % 2 == 0) {
            squaredErrorControlSet = this->squaredErrorControlSet();
            errorControlSet.push_back(squaredErrorControlSet);
            if (errorControlSet.size() > sampleCount) {
                double sumErrorControlSet = 0.0;
                for (int index = 0; index < sampleCount; index++) {
                    sumErrorControlSet += errorControlSet[index];
                }
                avgErrorControlSet.push_back(sumErrorControlSet);
            }
            this->fitLine(avgErrorControlSet, &slope, &offset);
        }
        condition1 = squaredError <= maxRMSE || squaredErrorControlSet <= maxRMSE;
        condition2 = epoch++ > maxEpoch;
        condition3 = slope > 0;
    } while (!condition1 && !condition2 && !condition3);
    this->epoch = epoch;
    this->errorTrainingSet = squaredError;
    this->errorControlSet = squaredErrorControlSet;
    this->success = condition1;
    return condition1;
}

/**
 * Returns the output of the model based on the input
 * 
 * @param double input[column] 
 *      The row of the training data 
 * @return double output[column]
 *      The output of the model
 */
vector<double> NeuralNetwork::feedforward(vector<double> input) {
    for (int node = 0; node < this->nodeCount[0]; node++) {
        this->nodeValue[0][node] = input[node];
    }
    for (int layer = 1; layer < this->nodeCount.size(); layer++) {
        for (int node = 0; node < this->nodeCount[layer]; node++) {
            double weightedSum = 0.0;
            for (int previousNode = 0; previousNode < this->nodeCount[layer-1]; previousNode++) {
                weightedSum += this->nodeValue[layer-1][previousNode] * this->nodeWeight[layer-1][previousNode][node];
            }
            weightedSum -= this->nodeThreshold[layer][node];
            weightedSum = this->activation(weightedSum);
            this->nodeValue[layer][node] = weightedSum;
        }
    }
    return this->nodeValue[this->nodeCount.size()-1];
}

/**
 * Corrects the weights and threshold of the model based on the output and desired output using backpropagation algorithm
 * 
 * @param double output[column]
 *      The output of the model
 * @param double targetOutput[column]
 *      The desired output of the model
 */
void NeuralNetwork::backpropagate(vector<double> output, vector<double> targetOutput) {
    vector<vector<double>> errorGradient = this->nodeValue;
    for (int layer = this->nodeCount.size()-1; layer > 0; layer--) {
        for (int node = 0; node < this->nodeCount[layer]; node++) {
            if (layer == this->nodeCount.size()-1) {
                double error = output[node] - targetOutput[node];
                errorGradient[layer][node] = this->derivativeActivation(output[node]) * error;
            }
            else {
                double productSum = 0.0;
                for (int nextNode = 0; nextNode < this->nodeCount[layer+1]; nextNode++) {
                    productSum += errorGradient[layer+1][nextNode] * this->nodeWeight[layer][node][nextNode];
                }
                errorGradient[layer][node] = this->derivativeActivation(this->nodeValue[layer][node]) * productSum;
            }
            for(int previousNode = 0; previousNode < this->nodeCount[layer-1]; previousNode++) {
                double weightCorrection = this->learningRate * this->nodeValue[layer-1][previousNode] * errorGradient[layer][node];
                this->nodeWeight[layer-1][previousNode][node] = this->nodeWeight[layer-1][previousNode][node] + weightCorrection + this->momentum * this->nodeWeightCorrection[layer][node];
                this->nodeWeightCorrection[layer][node] = weightCorrection;
            }
        }
    }
}

/**
 * Calculates sigmoid activation function on the value
 * 
 * @param double value
 *      The value that will be used
 * @return double
 *      The output of the sigmoid activation function
 */
double NeuralNetwork::activation(double value) {
    return 1.0 / (1.0 + exp(value - (value * 2)));
}

/**
 * Calculates derivative sigmoid activation function on the value
 * 
 * @param double value
 *      The value that will be used
 * @return double
 *      The output of the derivative sigmoid activation function
 */
double NeuralNetwork::derivativeActivation(double value) {
    return value * (1.0 - value);
}

/**
 * Generates random value for the weightCorrection
 * 
 * @return double
 *      The generated random value
 */
double NeuralNetwork::generateRandomWeight() {
    return ((rand() / 1000) - 0.5) / 2;
}

/**
 * ...
 * 
 * @param double trainingData[row][column]
 *      ...
 * @return double
 *      ...
 */
double NeuralNetwork::squaredErrorEpoch(vector<vector<double>> trainingData) {
    double rmse = 0.0;
    for (int row = 0; row < trainingData.size(); row++) {
        vector<double> input;
        vector<double> targetOutput;
        for (int column = 0; column < this->nodeCount[0]; column++) {
            input.push_back(trainingData[row][column]);
        }
        for (int column = this->nodeCount[0]; column < this->nodeCount[0] + this->nodeCount[this->nodeCount.size()-1]; column++) {
            targetOutput.push_back(trainingData[row][column]);
        }

        rmse += this->squaredError(input, targetOutput);
    }
    rmse = rmse / trainingData.size();
    return sqrt(rmse);
}

/**
 * ...
 * 
 * @param input[column]
 *      ...
 * @param targetOutput[column]
 *      ...
 * @return double
 *      ...
 */
double NeuralNetwork::squaredError(vector<double> input, vector<double> targetOutput) {
    vector<double> output = this->feedforward(input);
    double rmse = 0.0;
    for (int node = 0; node < input.size(); node++) {
        rmse += output[node] - targetOutput[node];
    }
    return rmse;
}

/**
 * ...
 * 
 * @return double
 *      ...
 */
double NeuralNetwork::squaredErrorControlSet() {
    if (this->controlData.size() == 0) {
        return 1.0;
    }
    double rmse = 0.0;
    for (int row = 0; row < this->controlData.size(); row++) {
        vector<double> controlInput;
        vector<double> controlOutput;

        for (int column = 0; column < this->nodeCount[0]; column++) {
            controlInput.push_back(this->controlData[row][column]);
        }

        for (int column = this->nodeCount[0]; column < this->nodeCount[0] + this->nodeCount[this->nodeCount.size()-1]; column++) {
            controlOutput.push_back(this->controlData[row][column]);
        }
        rmse += this->squaredError(controlInput, controlOutput);
    }
    return rmse;
}

/**
 * ...
 * 
 * @param avgErrorControlSet[index]
 *      ...
 * @param double *slope
 *      ...
 * @param double *offset
 *      ...
 */
void NeuralNetwork::fitLine(vector<double> avgErrorControlSet, double *slope, double *offset) {
    if (avgErrorControlSet.size() > 1) {
        double sumX = 0.0;
        double sumX2 = 0.0;
        double sumY = 0.0;
        double sumXY = 0.0;
        for (int index = 0; index < avgErrorControlSet.size(); index++) {
            sumX += index;
            sumY += avgErrorControlSet[index];
            sumX2 += index * index;
            sumXY += index * avgErrorControlSet[index];
        }
        *slope = (avgErrorControlSet.size() * sumXY - sumX * sumY) / (avgErrorControlSet.size() * sumX2 - sumX * sumX);
        *offset = (sumY * sumX2 - sumX * sumXY) / (avgErrorControlSet.size() * sumX2 - sumX * sumX);
    } 
    else {
        *slope = 0.0;
        *offset = 0.0;
    }
}