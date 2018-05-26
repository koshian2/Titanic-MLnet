using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;

namespace Titanic
{
    class Program
    {
        //データのパス
        static string trainSetPath = Path.Combine(Environment.CurrentDirectory, "data", "titanic_train.csv");
        static string crossValidationSetPath = Path.Combine(Environment.CurrentDirectory, "data", "titanic_cv.csv");
        static string testSetPath = Path.Combine(Environment.CurrentDirectory, "data", "titanic_test.csv");

        static void Main(string[] args)
        {
            //C#7.1以降ならasync MainでOK
            Train().Wait();
        }

        static async Task Train()
        {
            //パイプラインの作成
            var pipeline = new LearningPipeline();

            //訓練データの読み込み
            var trainingSets = new TextLoader<TitanicData>(trainSetPath, useHeader: true, separator: ",");
            pipeline.Add(trainingSets);

            //年齢が欠損値の行を捨てる
            pipeline.Add(new MissingValuesRowDropper()
            {
                Column = new string[] { "Age" }
            });

            //数値でない変数をOneHotVectorにする
            pipeline.Add(new CategoricalOneHotVectorizer("Sex", "Embarked"));

            //モデルに使う変数を結合する
            pipeline.Add(new ColumnConcatenator("Features",
                "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"));

            //交差検証データ
            var cvSets = new TextLoader<TitanicData>(crossValidationSetPath, useHeader: true, separator: ",");
            //グリッドサーチ用
            var n_trees = new int[] { 2, 4, 8, 16, 32, 64, 128};
            var n_leaves = new int[] { 2, 4, 8, 16, 32, 64, 128};
            //最も良い精度
            var bestf1 = 0.0;
            //最も良い分類器
            FastForestBinaryClassifier bestClassifier = null;
            //二値分類の評価
            var evaluator = new BinaryClassificationEvaluator();
            foreach (var nt in n_trees)
            {
                foreach(var nl in n_leaves)
                {
                    //ランダムフォレストで二値分類
                    var classifier = new FastForestBinaryClassifier()
                    {
                        NumTrees = nt,
                        NumLeaves = nl
                    };
                    pipeline.Add(classifier);
                    //訓練
                    var model = pipeline.Train<TitanicData, TitanicPrediction>();

                    //F1スコア
                    var metrics = evaluator.Evaluate(model, cvSets);
                    Console.WriteLine($"#tree = {nt}, #leaf = {nl}, cv_f1={metrics.F1Score}");
                    if(!double.IsNaN(metrics.F1Score) && metrics.F1Score > bestf1)
                    {
                        Console.WriteLine($"[!]Classifier Updated {bestf1} -> {metrics.F1Score} / nt : {nt}, nl : {nl}");
                        bestf1 = metrics.F1Score;
                        bestClassifier = classifier;
                    }

                    //パイプラインから一旦分類器削除
                    pipeline.Remove(classifier);
                }
            }

            //グリッドサーチの結果から最も良いモデルを選択してパイプラインに追加
            pipeline.Add(bestClassifier);

            //訓練
            var bestModel = pipeline.Train<TitanicData, TitanicPrediction>();
            //訓練誤差
            var trainMetrics = evaluator.Evaluate(bestModel, trainingSets);
            //交差検証誤差
            var cvMetrics = evaluator.Evaluate(bestModel, cvSets);
            //テストデータ
            var testSets = new TextLoader<TitanicData>(testSetPath, useHeader: true, separator: ",");
            //テスト誤差
            var testMetrics = evaluator.Evaluate(bestModel, testSets);

            //モデルの保存
            await bestModel.WriteAsync("model.zip");

            //結果表示
            Console.WriteLine("### Result ###");
            Console.WriteLine("- Selected Classifier");
            Console.WriteLine($"NumTree={bestClassifier.NumTrees}, NumLeaves={bestClassifier.NumLeaves}");
            Console.WriteLine("- Trian Sets");
            Console.WriteLine($"Accuracy={trainMetrics.Accuracy:P2}, " +
                $"Precision={trainMetrics.PositivePrecision:P2}, " +
                $"Recall={trainMetrics.PositiveRecall:P2}, " +
                $"F1score={trainMetrics.F1Score:P2}");
            Console.WriteLine("- Cross Validation Sets");
            Console.WriteLine($"Accuracy={cvMetrics.Accuracy:P2}, " +
                $"Precision={cvMetrics.PositivePrecision:P2}, " +
                $"Recall={cvMetrics.PositiveRecall:P2}, " +
                $"F1score={cvMetrics.F1Score:P2}");
            Console.WriteLine("- Test Sets");
            Console.WriteLine($"Accuracy={testMetrics.Accuracy:P2}, " +
                $"Precision={testMetrics.PositivePrecision:P2}, " +
                $"Recall={testMetrics.PositiveRecall:P2}, " +
                $"F1score={testMetrics.F1Score:P2}");
            /*
            ### Result ###
            - Selected Classifier
            NumTree=4, NumLeaves=32
            - Trian Sets
            Accuracy=83.45%, Precision=83.78%, Recall=72.94%, F1score=77.99%
            - Cross Validation Sets
            Accuracy=80.28%, Precision=86.67%, Recall=63.93%, F1score=73.58%
            - Test Sets
            Accuracy=86.58%, Precision=86.79%, Recall=77.97%, F1score=82.14%
            */
        }
    }
}
