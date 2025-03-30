using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_Network
{
    static class MNIST
    {
        public static List<int> Labels;
        public static List<double[]> Images;
        public static List<double[,]> ImageArray;

        public static int totalImages;

        public static void Init(List<int> labels, List<double[,]> images, List<double[]> imagesOneD)
        {
            Labels = labels;
            ImageArray = images;
            Images = imagesOneD;
            totalImages = images.Count;
        }
    }
}
