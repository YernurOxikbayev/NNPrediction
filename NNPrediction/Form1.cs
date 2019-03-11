using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace NNPrediction
{
    public partial class Form1 : Form
    {
        
        int layersNumber, nN = 0, maxEpoch;
        int immersionsDepth;
        int trainSetsNumber;
        double[] predictionResults;
        double max = -300.0, min = 300.0, initEvData;
        double[] inputSequence, evalData;
        bool stop = false;
        public double error = 1;
        static Random rnd = new Random();
        Layer[] Net;

        public class Neuron
        {
            public double [] w;
            public double s;
            public double x;
            public double b;
            public int relNumber;
            public Neuron()
            {
            }
            public void weight()
            {
                w = new double[relNumber];
                for (int i = 0; i < relNumber; i++)
                    w[i] = rnd.Next(200) / 1000.0 - 0.1;
                s = 0;
                x = 0;
                b = 0;
                return;
            }
        }

        public class Layer
        {
            public int neuronsNumber;
            public Neuron []Neuron;
            public Layer()
            {
                neuronsNumber = 0;
            }
            public void createNeurons()
            { 
                Neuron = new Neuron[neuronsNumber];
                for (int i = 0; i < neuronsNumber; i++)
                {
                    Neuron[i] = new Neuron();
                }
                return;
            }
        }
                
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            button2.Visible = true;
            layersNumber = Convert.ToInt32(textBox1.Text) + 2;// скрытых слоев
            int neuronsNumber = Convert.ToInt32(textBox2.Text);// нейронов слое
            immersionsDepth = Convert.ToInt32(textBox7.Text);// глубина погружения
            inputSequence = ReadFromFile("R0123.txt");
            trainSetsNumber = (int)(0.8 * inputSequence.Length);// обучающий набор 80%
            predictionResults = new double[inputSequence.Length - trainSetsNumber - immersionsDepth];// тестовый набор
            evalData = new double[predictionResults.Length];
            for (int i = 0; i < evalData.Length; i++)
            {
                evalData[i] = inputSequence[trainSetsNumber + i + immersionsDepth];
            }
            maxEpoch = Convert.ToInt32(textBox3.Text);// кол-во эпох
            Net = new Layer[layersNumber];

            for (int i = 0; i < layersNumber; i++)
            {
                Net[i] = new Layer();
            }

            Net[0].neuronsNumber = immersionsDepth; // ввод чисел
            Net[0].createNeurons();

           
            for (int i = 1; i < layersNumber; i++ )
            {
                if (i == (layersNumber - 1)) Net[i].neuronsNumber = 1;
                else Net[i].neuronsNumber = neuronsNumber;
                Net[i].createNeurons();
                for (int j = 0; j < Net[i].neuronsNumber; j++)
                {
                    Net[i].Neuron[j].relNumber = Net[i - 1].neuronsNumber; //кол-во связей нейрона=кол-ву нейронов предыдущего слоя
                    Net[i].Neuron[j].weight();
                }
            }
        }

        private double[] ReadFromFile(string fileName)
        {
            string line = File.ReadLines(fileName).First();
            string[] inputV;
            inputV = line.Split(null);
            int length = inputV.Length;
            double[] result = new double[length];

            for (int i = 0; i < length; i++)
            {
                result[i] = double.Parse(inputV[i], CultureInfo.InvariantCulture);

                if (result[i] > max)
                    max = result[i];
                if (result[i] < min)
                    min = result[i];
            }

            if (checkBox1.Checked)
            {                
                double[] diffVals = new double[length];
                initEvData = result[trainSetsNumber + immersionsDepth];
                for (int i = 1; i < length; i++)
                {
                    diffVals[i] = result[i] - result[i - 1];

                    if (diffVals[i] > max)
                        max = diffVals[i];
                    if (diffVals[i] < min)
                        min = diffVals[i];
                }
                result = diffVals;
            }

            for (int i = 0; i < length; i++)
            {
                result[i] = (result[i] - min) / (max - min);
            }

            return result;
        }

        void Result()
        {
            double Sum;
            for (int i = 1; i < layersNumber; i++ )
            {
                for (int j = 0; j < Net[i].neuronsNumber; j++)
                {
                    Sum = 0;
                    for (int k = 0; k < Net[i].Neuron[j].relNumber; k++)
                        Sum += (Net[i - 1].Neuron[k].x * Net[i].Neuron[j].w[k]);

                    Sum += Net[i].Neuron[j].s;
                    Net[i].Neuron[j].x = 1.0 / (1.0 + Math.Pow(Math.E, -Sum)); ;       //функция активации
                }
            }
        }

        void CalcError()
        {
            double temp = 0;
            button3.Visible = true;
            Net[layersNumber - 1].Neuron[0].b = Net[layersNumber - 1].Neuron[0].x
                * (1 - Net[layersNumber - 1].Neuron[0].x) * (inputSequence[nN + immersionsDepth]
                - Net[layersNumber - 1].Neuron[0].x);
            //по методу обратного распространения ошибки
            for (int i = layersNumber - 2; i > 0; i--)
                for (int j = 0; j < Net[i].neuronsNumber; j++)   //коррекция веса следующего слоя
                {
                    temp = 0;
                    for (int k = 0; k < Net[i + 1].neuronsNumber; k++)
                        temp += (Net[i + 1].Neuron[k].b * Net[i + 1].Neuron[k].w[j]);
                    temp *= (Net[i].Neuron[j].x * (1 - Net[i].Neuron[j].x));
                    Net[i].Neuron[j].b = temp;
                }
            for (int i = layersNumber - 2; i >= 0; i--)
                for (int j = 0; j < Net[i].neuronsNumber; j++)
                    for (int k = 0; k < Net[i + 1].neuronsNumber; k++)
                    {
                        Net[i + 1].Neuron[k].w[j] += (0.8 * Net[i + 1].Neuron[k].b * Net[i].Neuron[j].x);
                    }
            for (int i = layersNumber - 1; i > 0; i--)
                for (int k = 0; k < Net[i].neuronsNumber; k++)
                    Net[i].Neuron[k].s += (0.8 * Net[i].Neuron[k].b);
        }
        //обучает сеть
        private void button2_Click(object sender, EventArgs e)
        {
            double temp;
            int epoch = 0;
            stop = false;
            while (!stop && epoch < maxEpoch)
            {
                error = 0;
                for (nN = 0; nN < trainSetsNumber - immersionsDepth; nN++)
                {
                    for (int i = 0; i < Net[0].neuronsNumber; i++)
                    {

                        Net[0].Neuron[i].x = inputSequence[nN + i]; //подаем на вход
                    }
                    Result();
                    CalcError();
                    temp = inputSequence[nN + immersionsDepth] - Net[layersNumber - 1].Neuron[0].x;// разн между
                    if (temp < 0) temp *= -1;
                    error += temp;
                }
                error /= (trainSetsNumber - immersionsDepth);
                label4.Text = Convert.ToString(error);
                Application.DoEvents();
                epoch++;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            stop = true;
        }
        //тестирует сеть
        private void button4_Click(object sender, EventArgs e)
        {
            double sum = 0.0;
            double devSum = 0.0;
            int resLength = predictionResults.Length;
            for (int i = 0; i < resLength; i++)
            {
                for (int j = 0; j < immersionsDepth; j++)
                {
                    Net[0].Neuron[j].x = inputSequence[i + trainSetsNumber + j];
                }
                Result();
                predictionResults[i] = Net[layersNumber - 1].Neuron[0].x;

                label6.Text = Convert.ToString(predictionResults[resLength - 1] * (max - min) + min);

                sum += 100 * (1 - Math.Abs(predictionResults[i] - evalData[i]) / evalData[i]);
                devSum += ((predictionResults[i] * (max - min) + min) - (evalData[i] * (max - min) + min))
                    * ((predictionResults[i] * (max - min) + min) - (evalData[i] * (max - min) + min)) / resLength;
                                
               
            }

            if (checkBox1.Checked)
            {                
                evalData[0] = initEvData;
                predictionResults[0] = initEvData;

                sum = 100 * (1 - Math.Abs(predictionResults[0] - evalData[0]) / evalData[0]);
                devSum = (predictionResults[0] - evalData[0]) * (predictionResults[0] - evalData[0]) / resLength; ;
                for (int i = 1; i < evalData.Length; i++)
                {
                    evalData[i] = (evalData[i] * (max - min) + min) + evalData[i - 1];
                    predictionResults[i] = (predictionResults[i] * (max - min) + min) + predictionResults[i - 1];

                    sum += 100 * (1 - Math.Abs(predictionResults[i] - evalData[i]) / evalData[i]);
                    devSum += (predictionResults[i] - evalData[i]) * (predictionResults[i] - evalData[i]) / resLength;
                }
            }

            textBox5.Text = Convert.ToString(sum / resLength);
            textBox6.Text = Convert.ToString(Math.Sqrt(devSum));
        }


    }
}
