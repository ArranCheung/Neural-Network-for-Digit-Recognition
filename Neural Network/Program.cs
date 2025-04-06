using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Neural_Network
{
    class Network
    {
        private int NumInputNeurons;
        private int NumHiddenNeurons;
        private int NumOutputNeurons;

        private double[,] InputHiddenWeights;
        private double[,] OutputHiddenWeights;

        private double[] HiddenNeuronBias;
        private double[] OutputNeuronBias;

        private double[] HiddenActivation;
        private double[] OutputActivation;

        private double[] Input;

        public double CostValue;
        public double LearningCoefficient = 2.4;

        public int IntendedOutput;

        public Network(int InSize, int HidSize, int OutSize)
        {
            NumInputNeurons = InSize; NumHiddenNeurons = HidSize; NumOutputNeurons = OutSize;

            InputHiddenWeights = new double[InSize, HidSize];
            OutputHiddenWeights = new double[HidSize, OutSize];

            HiddenNeuronBias = new double[HidSize];
            OutputNeuronBias = new double[OutSize];

            HiddenActivation = new double[HidSize];
            OutputActivation = new double[OutSize];

            InitRandWeight(new List<double[,]> { InputHiddenWeights, OutputHiddenWeights });
            InitRandBias();
        }

        public void InitRandWeight(List<double[,]> arrays)
        {
            Random rand = new Random();

            foreach (var array in arrays)
            {
                for (int i = 0; i < array.GetLength(0); i++)
                {
                    for (int j = 0; j < array.GetLength(1); j++)
                    {
                        array[i, j] = rand.NextDouble();
                    }
                }
            }
        }

        public void InitRandBias()
        {
            Random rand = new Random();

            foreach (var item in new List<double[]> { HiddenNeuronBias, OutputNeuronBias })
            {
                for (int i = 0; i < item.Count(); i++)
                {
                    item[i] = rand.NextDouble();
                }
            }
        }

        public double[] ForwardPropagation(double[] input)
        {
            double[] HiddenOuputValues = new double[NumHiddenNeurons];

            for (int i = 0; i < InputHiddenWeights.GetLength(1); i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    double value = 0;
                    for (int k = 0; k < InputHiddenWeights.GetLength(0); k++)
                    {
                        value += InputHiddenWeights[k, i] * input[k];
                    }
                    value = SigmoidFunc(value - HiddenNeuronBias[i]);
                    HiddenOuputValues[i] = value;
                }
            }
            HiddenActivation = HiddenOuputValues;
            double[] OutputValues = new double[NumOutputNeurons];

            for (int i = 0; i < OutputHiddenWeights.GetLength(1); i++)
            {
                for (int j = 0; j < 1; j++)
                {
                    double value = 0;
                    for (int k = 0; k < OutputHiddenWeights.GetLength(0); k++)
                    {
                        value += OutputHiddenWeights[k, i] * HiddenActivation[i];
                    }
                    value = SigmoidFunc(value - OutputNeuronBias[i]);
                    OutputValues[i] = value;
                }
            }
            OutputActivation = OutputValues;

            double[] SMout = SoftMax(OutputValues);
            return SMout;
        }

        public void BackPropogation(int IntendedOutput, double[] input)
        {
            double[] OutputError = new double[OutputActivation.Length];

            for (int NeuronPerOutput = 0; NeuronPerOutput < OutputActivation.Length; NeuronPerOutput++)
            {
                double zL = OutputNeuronBias[NeuronPerOutput];
                for (int i = 0; i < HiddenActivation.Length; i++)
                {
                    zL += (OutputHiddenWeights[i, NeuronPerOutput] * HiddenActivation[i]);
                }
                OutputError[NeuronPerOutput] = zL;
            }

            // Gradient descent for the weights of the arcs
            for (int i = 0; i < HiddenActivation.Length; i++)
            {
                for (int j = 0; j < OutputActivation.Length; j++)
                {
                    double y = (IntendedOutput == j) ? 1 : 0;
                    double dCdW = (HiddenActivation[i]) * (DerivativeSigmoidFunc(OutputError[j])) * (2 * (OutputActivation[j] - y));

                    OutputHiddenWeights[i, j] -= LearningCoefficient * dCdW;
                }
            }
        }

        public double SigmoidFunc(double InputValue) => (1d / (1 + Math.Exp(0 - InputValue)));

        public double DerivativeSigmoidFunc(double InputValue)
        {
            double Sig = SigmoidFunc(InputValue);
            return Sig * (1 - Sig);
        }

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

        public double Cost(int DesiredValue, double[] Probabilities)
        {
            int OutputValue = MostLikely(Probabilities);
            double sum = 0;

            for (int i = 0; i < Probabilities.Length; i++)
            {
                double dif = Probabilities[i];

                if (i == OutputValue)
                {
                    dif -= 1;
                }

                sum += (Math.Pow(dif, 2));
            }

            return sum;
        }

        public int MostLikely(double[] Probabilities)
        {
            List<double> P = Probabilities.ToList();
            double max = Probabilities.Max();
            int index = P.IndexOf(max);

            return index;
        }

        public void SetCostValue(double value)
        {
            CostValue = value;
        }
    }


    internal class Program
    {
        static List<dynamic> ReadImages()
        {
            List<double[,]> Images = new List<double[,]>();
            List<double[]> ImageData = new List<double[]>();

            using (BinaryReader Br = new BinaryReader(File.OpenRead("C:\\Users\\arran\\source\\repos\\Neural Network\\Neural Network\\MNIST Labelled Dataset\\t10k-images.idx3-ubyte")))
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
                            ImageArray[rows, cols] = Convert.ToDouble(image[i]);
                            ImageOneD[i] = Convert.ToDouble(image[i]);

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

        static List<int> ReadLabels()
        {
            List<int> Labels = new List<int>();

            using (BinaryReader Br = new BinaryReader(File.OpenRead("C:\\Users\\arran\\source\\repos\\Neural Network\\Neural Network\\MNIST Labelled Dataset\\t10k-labels.idx1-ubyte")))
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

        static void WriteImage(double[,] Matrix)
        {
            for (int i = 0; i < Matrix.GetLength(0); i++)
            {
                for (int j = 0; j < Matrix.GetLength(1); j++)
                {
                    if (Matrix[i, j] > 60)
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
        }

        static void Main(string[] args)
        {
            // Reading Labels and Images
            List<int> labels = ReadLabels();
            List<dynamic> images = ReadImages();
            MNIST.Init(labels, images[0], images[1]);

            // Instantiating a new Network
            Network NeuralNetwork = new Network(784, 128, 10);

            double totalCost = 0;

            for (int i = 0; i < MNIST.totalImages; i++)
            {
                double[] Image = MNIST.Images[i];

                double[] Val = NeuralNetwork.ForwardPropagation(Image);

                NeuralNetwork.BackPropogation(MNIST.Labels[i], Image);

                totalCost += NeuralNetwork.Cost(MNIST.Labels[i], Val);
            }

            double averageCost = totalCost / MNIST.totalImages;
            NeuralNetwork.SetCostValue(averageCost);
            Console.WriteLine(averageCost);

            for (int i = 0; i < 5; i++)
            {
                Random rand = new Random();
                int index = rand.Next(0, 1000);

                double[] Val = NeuralNetwork.ForwardPropagation(MNIST.Images[index]);
                Console.WriteLine(NeuralNetwork.MostLikely(Val));
            }


            NeuralNetwork.InitRandBias();

            Console.ReadKey();
        }
    }
}
