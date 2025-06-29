using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Neural_Network
{
    class Network
    {
        private static Random rand = new Random();

        private int NumInputNeurons;
        private int NumHiddenNeurons;
        private int NumOutputNeurons;
        private int NumHiddenLayers;

        private double[,] InputHiddenWeights;
        private double[,] OutputHiddenWeights;

        private double[] HiddenNeuronBias;
        private double[] OutputNeuronBias;

        private double[] HiddenActivation;
        private double[] OutputActivation;

        private double[] Input;

        public double CostValue;
        public double LearningCoefficient = 0.008;
        public int IntendedOutput;

        public List<double[]> layersActivation;
        public List<double[,]> layersWeights;
        public List<double[]> layersBias;

        public Network(int InSize, int HidSize, int NumHidLayer, int OutSize)
        {
            NumInputNeurons = InSize; NumHiddenNeurons = HidSize; NumOutputNeurons = OutSize; NumHiddenLayers = NumHidLayer;

            layersWeights = new List<double[,]>();
            InputHiddenWeights = new double[InSize, HidSize];
            OutputHiddenWeights = new double[HidSize, OutSize];
            layersWeights.Add(InputHiddenWeights);

            layersBias = new List<double[]>();
            HiddenNeuronBias = new double[HidSize];
            OutputNeuronBias = new double[OutSize];
            layersBias.Add(HiddenNeuronBias);

            layersActivation = new List<double[]> { Input };
            HiddenActivation = new double[HidSize];
            OutputActivation = new double[OutSize];
            layersActivation.Add(HiddenActivation);

            if (NumHidLayer > 1)
            {
                for (int i = 0; i < NumHidLayer - 1; i++)
                {
                    double[,] MiddleHiddenWeights = new double[HidSize, HidSize];
                    layersWeights.Add(MiddleHiddenWeights);

                    double[] MiddleNeuronBias = new double[HidSize];
                    layersBias.Add(MiddleNeuronBias);

                    double[] MiddelActivation = new double[HidSize];
                    layersActivation.Add(MiddelActivation);
                }
            }
            layersWeights.Add(OutputHiddenWeights);
            layersBias.Add(OutputNeuronBias);
            layersActivation.Add(OutputActivation);

            InitRandBias();
            InitRandWeight(layersWeights);
            Console.WriteLine("Weights and biases loaded");
        }

        public void InitRandWeight(List<double[,]> arrays)
        {
            foreach (var array in arrays)
            {
                for (int i = 0; i < array.GetLength(0); i++)
                {
                    for (int j = 0; j < array.GetLength(1); j++)
                    {
                        array[i, j] = rand.NextDouble() * 2 - 1;
                    }
                }
            }
            WriteWeights();
        }
        public void InitRandBias()
        {
            foreach (var item in new List<double[]> { HiddenNeuronBias, OutputNeuronBias })
            {
                for (int i = 0; i < item.Count(); i++)
                {
                    item[i] = rand.NextDouble();
                }
            }
            WriteBias();
        }
        public void ReadWeights()
        {
            using (StreamReader sr = new StreamReader("Weights.txt"))
            {
                int row = 0;

                int target = 0;
                List<double[,]> targets = new List<double[,]> { InputHiddenWeights, OutputHiddenWeights };

                while (!sr.EndOfStream)
                {
                    var words = sr.ReadLine();

                    if (words != "")
                    {
                        if (words == "Output Hidden Weights")
                        {
                            row = 0; target++;
                        }

                        if (words != "Input Hidden Weights" && words != "Output Hidden Weights" && words != "")
                        {
                            var split = words.Split(' ');

                            for (int i = 0; i < split.Length; i++)
                            {
                                if (split[i] != "")
                                {
                                    targets[target][row, i] = Convert.ToDouble(split[i]);
                                }
                            }

                            row++;
                        }
                    }
                }
            }
        }
        public void WriteWeights()
        {
            using (StreamWriter sw = new StreamWriter("Weights.txt"))
            {
                sw.WriteLine("Input Hidden Weights");
                for (int i = 0; i < InputHiddenWeights.GetLength(0); i++)
                {
                    for (int j = 0; j < InputHiddenWeights.GetLength(1); j++)
                    {
                        sw.Write($"{InputHiddenWeights[i, j]} ");
                    }
                    sw.WriteLine();
                }

                sw.WriteLine("Output Hidden Weights");
                for (int i = 0; i < OutputHiddenWeights.GetLength(0); i++)
                {
                    for (int j = 0; j < OutputHiddenWeights.GetLength(1); j++)
                    {
                        sw.Write($"{OutputHiddenWeights[i, j]} ");
                    }
                    sw.WriteLine();
                }
            }
        }
        public void ReadBias()
        {
            using (StreamReader sr = new StreamReader("Bias.txt"))
            {
                bool hidden = true;
                while (!sr.EndOfStream)
                {
                    string words = sr.ReadLine();

                    if (words == "Output Neuron Bias") { hidden = false; }

                    if (words != "Output Neuron Bias" && words != "Hidden Neuron Bias")
                    {
                        int pos = 0;
                        foreach (var bias in words.Split(' '))
                        {
                            if (bias != "")
                            {
                                if (hidden)
                                {
                                    HiddenNeuronBias[pos] = Convert.ToDouble(bias);
                                }
                                else
                                {
                                    OutputNeuronBias[pos] = Convert.ToDouble(bias);
                                }
                                pos++;
                            }
                        }
                    }
                }
            }
        }
        public void WriteBias()
        {
            using (StreamWriter sw = new StreamWriter("Bias.txt"))
            {
                sw.WriteLine("Hidden Neuron Bias");
                foreach (var bias in HiddenNeuronBias)
                {
                    sw.Write($"{bias} ");
                }
                sw.WriteLine();
                sw.WriteLine("Output Neuron Bias");
                foreach (var bias in OutputNeuronBias)
                {
                    sw.Write($"{bias} ");
                }
            }
        }
        public void Save()
        {
            WriteWeights();
            WriteBias();
        }

        public double[] ForwardPropagation(double[] input, int ActualOutput)
        {
            IntendedOutput = ActualOutput;
            Input = input;
            layersActivation[0] = input;

            double[] OutputValues = new double[NumHiddenNeurons];

            // Want to repeat this block for the amount of hidden layers
            for (int layer = 0; layer < NumHiddenLayers; layer++)
            {
                OutputValues = new double[NumHiddenNeurons];
                var currentLayer = layersWeights[layer];

                Parallel.For(0, currentLayer.GetLength(1), i =>
                {
                    double value = 0;
                    for (int k = 0; k < currentLayer.GetLength(0); k++)
                    {
                        value += currentLayer[k, i] * layersActivation[layer][k];
                    }

                    OutputValues[i] = ReLU(value + layersBias[layer][i]);
                });
                layersActivation[layer + 1] = OutputValues;
            }

            OutputValues = new double[NumOutputNeurons];

            var lastlayerweights = layersWeights.Last();
            Parallel.For(0, lastlayerweights.GetLength(1), i =>
            {
                double value = 0;
                for (int k = 0; k < lastlayerweights.GetLength(0); k++)
                {
                    value += lastlayerweights[k, i] * layersActivation[layersActivation.Count - 2][k];
                }

                OutputValues[i] = (value + OutputNeuronBias[i]);
            });

            double[] SMout = SoftMax(OutputValues);
            layersActivation[layersActivation.Count - 1] = SMout;

            return SMout;
        }
        public double HiddenDeltaErrorCalc(int layer, int activationIndex, int weightIndex, double activationFromPrevious) => (layersWeights[layersWeights.Count - 1 - layer][weightIndex, activationIndex]) * activationFromPrevious;
        public void BackPropogation(int target, double[] input)
        {
            double[] outputDeltas = new double[NumOutputNeurons];
            Parallel.For(0, NumOutputNeurons, OutNeuron =>
            {
                double y = (target == OutNeuron) ? 1 : 0;
                outputDeltas[OutNeuron] = (layersActivation.Last()[OutNeuron] - y);
            });

            List<double[]> layerHiddenDelta = new List<double[]> { outputDeltas };
            for (int i = 0; i < NumHiddenLayers; i++)
            {
                double[] hiddenDeltas = new double[NumHiddenNeurons];

                Parallel.For(0, NumHiddenNeurons, HiddenNeuron =>
                {
                    double a = layersActivation[layersActivation.Count - 2 - i][HiddenNeuron];
                    double sum = 0;
                    int neuronCount = (i == 0) ? NumOutputNeurons : NumHiddenNeurons;
                    for (int o = 0; o < neuronCount; o++)
                    {
                        sum += HiddenDeltaErrorCalc(i, o, HiddenNeuron, layerHiddenDelta[i][o]);
                    }
                    hiddenDeltas[HiddenNeuron] = sum * DerivativeReLu(a);
                });

                layerHiddenDelta.Add(hiddenDeltas);
            }

            for (int layer = 0; layer < NumHiddenLayers; layer++)
            {
                Parallel.For(0, NumHiddenNeurons, HiddenNeuron =>
                {
                    int compare = (layer == 0) ? NumOutputNeurons : NumHiddenNeurons;
                    for (int o = 0; o < compare; o++)
                    {
                        layersWeights[layersWeights.Count - 1 - layer][HiddenNeuron, o] -= LearningCoefficient * layersActivation[layersActivation.Count - 2 - layer][HiddenNeuron] * layerHiddenDelta[layer][o];
                    }

                    layersBias[layer][HiddenNeuron] -= LearningCoefficient * layerHiddenDelta[layerHiddenDelta.Count - 1 - layer][HiddenNeuron];
                });
            }

            Parallel.For(0, NumOutputNeurons, OutNeuron =>
            {
                OutputNeuronBias[OutNeuron] -= LearningCoefficient * outputDeltas[OutNeuron];
            });

            Parallel.For(0, NumInputNeurons, InNeuron =>
            {
                for (int HiddenNeuron = 0; HiddenNeuron < NumHiddenNeurons; HiddenNeuron++)
                {
                    InputHiddenWeights[InNeuron, HiddenNeuron] -= LearningCoefficient * input[InNeuron] * layerHiddenDelta.Last()[HiddenNeuron];
                }
            });
        }
        public double SigmoidFunc(double InputValue) => (1d / (1 + Math.Exp(0 - InputValue)));
        public double DerivativeSigmoidFunc(double InputValue)
        {
            double Sig = SigmoidFunc(InputValue);
            return Sig * (1 - Sig);
        }
        public double ReLU(double InputValue) => (0 > InputValue) ? 0 : InputValue;
        public double DerivativeReLu(double InputValue) => (0 < InputValue) ? 1 : 0;
        public double[] SoftMax(double[] InputValues)
        {
            double[] OutputProbabilities = new double[InputValues.Length];

            for (int i = 0; i < InputValues.Length; i++)
            {
                OutputProbabilities[i] = Math.Pow(Math.E, InputValues[i]);
            }

            double sum = OutputProbabilities.Sum();
            for (int i = 0; i < OutputProbabilities.Length; i++)
            {
                OutputProbabilities[i] = OutputProbabilities[i] / sum;
            }

            return OutputProbabilities;
        }

        public int TestCase(double[] Image, int Desired)
        {
            double[] Val = ForwardPropagation(Image, Desired);
            int guess = MostLikely(Val);

            Console.WriteLine($"Network's guess: {guess}");

            if (guess == Desired)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("Correct");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Incorrect");
                Console.ResetColor();
            }

            return guess;
        }
        public double CostCalc(double OutputValue, double Target) => (OutputValue - Target);
        public double Cost(int DesiredValue, double[] Probabilities)
        {
            double sum = 0;

            for (int i = 0; i < Probabilities.Length; i++)
            {
                double target = (i == DesiredValue) ? 1 : 0;

                sum += CostCalc(Probabilities[i], target);
            }

            return sum;
        }
        public void SetCostValue(double value)
        {
            CostValue = value;
        }
        public int MostLikely(double[] Probabilities)
        {
            List<double> P = Probabilities.ToList();
            double max = Probabilities.Max();
            int index = P.IndexOf(max);

            return index;
        }
    }


    internal class Program
    {
        static Random rand = new Random();
        static Network NeuralNetwork;

        static List<int> NewIncorrect = new List<int>();
        static List<int> PreviousWrong;

        static List<dynamic> ReadImages(string path)
        {
            List<double[,]> Images = new List<double[,]>();
            List<double[]> ImageData = new List<double[]>();

            using (BinaryReader Br = new BinaryReader(File.OpenRead(path)))
            {
                int MagicNumber = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);
                int NumImages = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);
                int RowsPerImage = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);
                int ColumnsPerImage = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);

                if (MagicNumber == 2051)
                {
                    for (int AmountOfImages = 0; AmountOfImages < NumImages; AmountOfImages++)
                    {
                        byte[] image = Br.ReadBytes(RowsPerImage * ColumnsPerImage).ToArray();

                        double[,] ImageArray = new double[RowsPerImage, ColumnsPerImage];
                        double[] ImageOneD = new double[ImageArray.Length];

                        int cols = 0; int rows = 0;
                        for (int i = 0; i < ImageArray.Length; i++)
                        {
                            ImageArray[rows, cols] = Convert.ToDouble(image[i]) / 255.0;
                            ImageOneD[i] = Convert.ToDouble(image[i]) / 255.0;

                            cols++;

                            if (cols == ColumnsPerImage)
                            {
                                cols = 0;
                                rows++;
                            }
                        }

                        Images.Add(ImageArray);
                        ImageData.Add(ImageOneD);
                    }
                }
            }

            return new List<dynamic> { Images, ImageData };
        }
        static List<int> ReadLabels(string path)
        {
            List<int> Labels = new List<int>();

            using (BinaryReader Br = new BinaryReader(File.OpenRead(path)))
            {
                int MagicNumber = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);

                if (MagicNumber == 2049)
                {
                    int NumLabels = BitConverter.ToInt32(Br.ReadBytes(4).Reverse().ToArray(), 0);

                    for (int i = 0; i < NumLabels; i++)
                    {
                        int label = Convert.ToInt32(Br.ReadByte());
                        Labels.Add(label);
                    }
                }

            }

            return Labels;
        }
        static void WriteImage(double[,] Matrix, int label = -1, bool display = false)
        {
            for (int i = 0; i < Matrix.GetLength(0); i++)
            {
                for (int j = 0; j < Matrix.GetLength(1); j++)
                {
                    if (Matrix[i, j] > 0.5)
                    {
                        Console.BackgroundColor = ConsoleColor.White;
                        Console.Write(" ");
                        Console.ResetColor();
                    }
                    else
                    {
                        Console.Write(" ");
                    }
                }
                Console.WriteLine();
            }

            if (display)
            {
                Console.WriteLine($"Value: {label}");
            }
        }
        static void WriteProb(double value)
        {
            File.WriteAllText("Stats.txt", value.ToString());
        }
        static void WriteIncorrect(List<int> value)
        {
            using (StreamWriter sw = new StreamWriter("Incorrect.txt"))
            {
                foreach (int item in value)
                {
                    sw.WriteLine(item);
                }
            }
        }
        static List<int> ReadIncorrect()
        {
            List<int> values = new List<int>();
            using (StreamReader sr = new StreamReader("Incorrect.txt"))
            {
                while (!sr.EndOfStream)
                {
                    values.Add(int.Parse(sr.ReadLine()));
                }
            }

            return values;
        }
        static void TrainingCycle(int BatchSize, int iterations, bool random = false, bool display = false)
        {
            for (int Iterations = 0; Iterations < iterations; Iterations++)
            {
                double totalCost = 0; int correct = 0;

                for (int i = 0; i < BatchSize; i++)
                {
                    int index = random ? rand.Next(0, MNIST.totalImages) : i;

                    // Select image from dataset
                    double[] Image = /*MNIST.Images[PreviousWrong[index]];*/ MNIST.Images[index];

                    // Display the image
                    //WriteImage(MNIST.ImageArray[index], MNIST.Labels[index]);

                    // Use network to determine the output
                    double[] Val = NeuralNetwork.ForwardPropagation(Image, MNIST.Labels[index]);
                    int guess = NeuralNetwork.MostLikely(Val);

                    if (display)
                    {
                        if (guess == MNIST.Labels[index])
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            correct++;
                        } // Update console colour to correctness
                        else
                        {
                            NewIncorrect.Add(index);
                        } // Write the incorrecly identified into to a file for further training

                        Console.Write(guess);
                        Console.ResetColor();
                    } // display the networks guess

                    // Train the network
                    NeuralNetwork.BackPropogation(MNIST.Labels[index], Image);

                    // Update the total cost for the network
                    totalCost += NeuralNetwork.Cost(MNIST.Labels[index], Val);

                    if (i % 100 == 0)
                    {
                        NeuralNetwork.Save();
                    } // Save network configuration periodically
                }

                // Calcuate the network cost
                double averageCost = totalCost / BatchSize;
                NeuralNetwork.SetCostValue(averageCost);

                // Calculate the accuracy of the model
                double perct = (double)correct / BatchSize;
                WriteProb(perct);

                if (display)
                {
                    Console.WriteLine();
                    Console.WriteLine(averageCost);

                    Console.ForegroundColor = (perct > 0.65) ? ConsoleColor.Green : ConsoleColor.White;
                    Console.ForegroundColor = (perct < 0.35) ? ConsoleColor.Red : ConsoleColor.White;

                    Console.WriteLine(perct);
                    Console.ResetColor();

                    Console.WriteLine();
                } // display the average cost ++ model accuracy
            }
        }

        static void InitaliseNetwork()
        {
            TrainingCycle(100, 10, true);
        }

        static void Main(string[] args)
        {
            // Lists for incorrect values identified
            PreviousWrong = ReadIncorrect();

            // Reading Labels and Images
            List<int> labels = ReadLabels("MNIST Labelled Dataset\\train-labels.idx1-ubyte");
            List<dynamic> images = ReadImages("MNIST Labelled Dataset\\train-images.idx3-ubyte");
            MNIST.Init(labels, images[0], images[1]);

            // Instantiating a new Network
            NeuralNetwork = new Network(784, 128, 2, 10);

            // Setup the networks weights correctly
            //InitaliseNetwork();

            // Load Test dataset
            List<int> testLabels = ReadLabels("MNIST Labelled Dataset\\t10k-labels.idx1-ubyte");
            List<dynamic> testImages = ReadImages("MNIST Labelled Dataset\\t10k-images.idx3-ubyte");
            MNIST.InitTest(testLabels, testImages[0], testImages[1]);

            // Show menu
            Menu();

            WriteIncorrect(NewIncorrect);
            NeuralNetwork.Save();
            Console.ReadKey();
        }

        static void Menu()
        {
            bool MenuAlive = true;
            while (MenuAlive)
            {
                Console.WriteLine();
                Console.WriteLine("1. Train network");
                Console.WriteLine("2. Test network");
                Console.WriteLine("3. Quit");

                Console.Write("Enter option: ");
                int option = int.Parse(Console.ReadLine());

                switch (option)
                {
                    case 1:
                        Console.WriteLine("Enter number of iterations");
                        int iterations = int.Parse(Console.ReadLine());

                        Console.WriteLine("Enter number of items per iteration");
                        int batch = int.Parse(Console.ReadLine());

                        TrainingCycle(batch, iterations, true, true);
                        break;
                    case 2:
                        Console.WriteLine("Test with random case (y/n)?");
                        string randCase = Console.ReadLine();
                        if (randCase.ToUpper() == "Y")
                        {
                            int index = rand.Next(0, MNIST.totalTestImages);
                            double[] image = MNIST.testImages[index];

                            WriteImage(MNIST.testImagesArray[index]);

                            NeuralNetwork.TestCase(image, MNIST.testLabels[index]);
                        }
                        else
                        {
                            // Put stuff here to let user draw out an image
                        }

                        break;
                    case 3:
                        MenuAlive = false;
                        break;
                }
            }
        }
    }
}