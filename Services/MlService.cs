using System.Globalization;
using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Services;

public class MlService
{
     private readonly MLContext _mlContext;

    public MlService()
    {
        _mlContext = new MLContext(seed: 42);
    }

    public ITransformer Train(string dataPath, out IDataView trainingData)
    {
        Console.WriteLine("Загружаем данные...");

        IDataView data = _mlContext.Data.LoadFromTextFile<PersonData>(
            dataPath,
            hasHeader: false,
            separatorChar: ',');
        
        Console.WriteLine(
            $"Count string: {_mlContext.Data.CreateEnumerable<PersonData>(
                data,
                reuseRowObject: false).Count()}");

        var split = _mlContext.Data.TrainTestSplit(
            data,
            testFraction: 0.2);
        trainingData=split.TrainSet;

        var pipeline =
            _mlContext.Transforms.Conversion.ConvertType(
                    "IsAthleteFloat",
                    nameof(PersonData.isAtlete))
                .Append(
                    _mlContext.Transforms.Concatenate(
                        "Features",
                        nameof(PersonData.Height),
                        nameof(PersonData.Weight),
                        "IsAthleteFloat",
                        nameof(PersonData.BMI)))
                .Append(
                    _mlContext.Transforms.Conversion.MapValueToKey(
                        "Label",
                        nameof(PersonData.result)))
                .Append(
                    _mlContext.MulticlassClassification.Trainers
                        .SdcaMaximumEntropy())
                .Append(
                    _mlContext.Transforms.Conversion.MapKeyToValue(
                        "PredictedLabel"));

        Console.WriteLine();
        Console.WriteLine("Начинаем обучение...");

        var model = pipeline.Fit(split.TrainSet);

        Console.WriteLine("TRAIN:");

        var trainPredictions =
            model.Transform(split.TrainSet);

        var trainMetrics =
            _mlContext.MulticlassClassification.Evaluate(
                trainPredictions);

        Console.WriteLine(
            $"Accuracy: {trainMetrics.MacroAccuracy:P2}");

        Evaluate(model, split.TestSet);
        Console.WriteLine("Обучение завершено!");

        Evaluate(model, split.TestSet);

        return model;
    }

    private void Evaluate(
        ITransformer model,
        IDataView testData)
    {
        var predictions = model.Transform(testData);

        var metrics =
            _mlContext.MulticlassClassification.Evaluate(
                predictions);

        Console.WriteLine();
        Console.WriteLine("Результаты:");

        Console.WriteLine(
            $"Accuracy: {metrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"MicroAccuracy: {metrics.MicroAccuracy:P2}");

        Console.WriteLine(
            $"LogLoss: {metrics.LogLoss:F4}");
        
    

        Console.WriteLine();
        Console.WriteLine("Confusion Matrix:");

        var matrix = metrics.ConfusionMatrix;

        for (int i = 0; i < matrix.NumberOfClasses; i++)
        {
            for (int j = 0; j < matrix.NumberOfClasses; j++)
            {
                Console.Write(
                    $"{matrix.Counts[i][j],8}");
            }

            Console.WriteLine();
        }
        Console.WriteLine();
        Console.WriteLine("Metrics by class:");

        for (int i = 0; i < matrix.NumberOfClasses; i++)
        {
            double truePositive=matrix.Counts[i][i];

            double falsePositive = 0;
            double falseNegative = 0;

            for (int j = 0; j < matrix.NumberOfClasses; j++)
            {
                if (j != i)
                {
                    falsePositive += matrix.Counts[i][j];
                    falseNegative += matrix.Counts[i][j];
                }
            }

            double precision = truePositive + falsePositive == 0
                ? 0
                : truePositive / (truePositive + falsePositive);
            double  recall =  truePositive + falseNegative == 0
                ? 0
                : truePositive / (truePositive + falseNegative);
            double f1 = precision + recall==0
                ? 0
                : 2 * precision  * recall / (precision + recall);
            Console.WriteLine();
            string className = i switch
            {
                0 => "Normal",
                1 => "Overweight",
                _ => $"Class {i}",
            };
            Console.WriteLine($"Class: {className}");
            Console.WriteLine($"Prescision: {precision:P2}");
            Console.WriteLine($"Recall: {recall:P2}");
            Console.WriteLine($"F1: {f1:P2}");
        }
    }
   

    public PersonPrediction Predict(
        ITransformer model,
        PersonData person)
    {
        var predictionEngine =
            _mlContext.Model.CreatePredictionEngine<
                PersonData,
                PersonPrediction>(model);

        return predictionEngine.Predict(person);
    }

    public void SaveModel(ITransformer model, IDataView trainingData, string modelPath)
    {
        _mlContext.Model.Save(model,trainingData.Schema, modelPath);
        Console.WriteLine();
        Console.WriteLine("Model saved!");
    }

    public ITransformer LoadModel(string modelPath)
    {
        Console.WriteLine();
        Console.WriteLine($"Load Model...{modelPath}");
        
        var model=_mlContext.Model.Load(modelPath, out _);
        Console.WriteLine("Model loaded!");
        return model;
    }
public void CompareFeaturesMultipleTimes(string dataPath)
{
    Console.WriteLine();
    Console.WriteLine("=================================");
    Console.WriteLine("   MULTIPLE FEATURE EXPERIMENT");
    Console.WriteLine("=================================");
    
    IDataView data =
        _mlContext.Data.LoadFromTextFile<PersonData>(
            dataPath,
            hasHeader: false,
            separatorChar: ',');

    const int experiments = 5;

    double[] model1 = new double[experiments];
    double[] model2 = new double[experiments];
    double[] model3 = new double[experiments];
    double[] model4 = new double[experiments];

    for (int i = 0; i < experiments; i++)
    {
        Console.WriteLine();
        Console.WriteLine($"========== EXPERIMENT {i + 1} ==========");
        
        var experimentMl =
            new MLContext(seed: 100 + i);

        var split =
            experimentMl.Data.TrainTestSplit(
                data,
                testFraction: 0.2);

        model1[i] = TestFeatureSet(
            experimentMl,
            "Weight",
            split.TrainSet,
            split.TestSet,
            new[]
            {
                nameof(PersonData.Weight)
            });

        model2[i] = TestFeatureSet(
            experimentMl,
            "Weight + IsAthlete",
            split.TrainSet,
            split.TestSet,
            new[]
            {
                nameof(PersonData.Weight),
                "IsAthleteFloat"
            });

        model3[i] = TestFeatureSet(
            experimentMl,
            "Weight + IsAthlete + BMI",
            split.TrainSet,
            split.TestSet,
            new[]
            {
                nameof(PersonData.Weight),
                "IsAthleteFloat",
                nameof(PersonData.BMI)
            });
        model4[i] = TestFeatureSet(
            experimentMl,
            "Weight+ IsAthlete + BMI + RandomFeature",
            split.TrainSet,
            split.TestSet,
            new[]
            {
                nameof(PersonData.Weight),
                "IsAthleteFloat",
                nameof(PersonData.BMI),
                nameof(PersonData.RandomFeature)
            });
    }

    Console.WriteLine();
    Console.WriteLine("=================================");
    Console.WriteLine("          FINAL RESULTS");
    Console.WriteLine("=================================");

    Console.WriteLine();
    Console.WriteLine(
        $"MODEL 1 Average: {model1.Average():P2}");

    Console.WriteLine(
        $"MODEL 2 Average: {model2.Average():P2}");

    Console.WriteLine(
        $"MODEL 3 Average: {model3.Average():P2}");
    
    Console.WriteLine(
        $"Model 4 Average: {model4.Average():P2}");
}
public void CompareFeatures(string dataPath)
{
    Console.WriteLine();
    Console.WriteLine("=================================");
    Console.WriteLine("       FEATURE EXPERIMENT");
    Console.WriteLine("=================================");

    IDataView data =
        _mlContext.Data.LoadFromTextFile<PersonData>(
            dataPath,
            hasHeader: false,
            separatorChar: ',');

    var split =
        _mlContext.Data.TrainTestSplit(
            data,
            testFraction: 0.2);

    TestFeatureSet(
        _mlContext,
        "MODEL 1: Weight",
        split.TrainSet,
        split.TestSet,
        new[]
        {
            nameof(PersonData.Weight)
        });

    TestFeatureSet(
        _mlContext,
        "MODEL 2: Weight + IsAthlete",
        split.TrainSet,
        split.TestSet,
        new[]
        {
            nameof(PersonData.Weight),
            "IsAthleteFloat"
        });

    TestFeatureSet(
        _mlContext,
        "MODEL 3: Weight + IsAthlete + BMI",
        split.TrainSet,
        split.TestSet,
        new[]
        {
            nameof(PersonData.Weight),
            "IsAthleteFloat",
            nameof(PersonData.BMI)
        });
}
    private double TestFeatureSet(
        MLContext mlContext,
        string name,
        IDataView trainData,
        IDataView testData,
        string[] features)
    {
        Console.WriteLine();
        Console.WriteLine($"--- {name} ---");

        var pipeline =
            mlContext.Transforms.Conversion.ConvertType(
                    "IsAthleteFloat",
                    nameof(PersonData.isAtlete))
                .Append(
                    mlContext.Transforms.Concatenate(
                        "Features",
                        features))
                .Append(
                    mlContext.Transforms.Conversion.MapValueToKey(
                        "Label",
                        nameof(PersonData.result)))
                .Append(
                    mlContext.MulticlassClassification.Trainers
                        .SdcaMaximumEntropy());

        Console.WriteLine("Обучение...");

        var model = pipeline.Fit(trainData);

        var trainPredictions = model.Transform(trainData);
        var trainMetrics = mlContext.MulticlassClassification.Evaluate(trainPredictions);
        
        var testPredictions = model.Transform(testData);
        var testMetrics = mlContext.MulticlassClassification.Evaluate(testPredictions);
        
        
        Console.WriteLine(
            $"TRAIN Accuracy: {trainMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"TEST  Accuracy: {testMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"Gap:             {(trainMetrics.MacroAccuracy - testMetrics.MacroAccuracy):P2}");

        return testMetrics.MacroAccuracy;
    }
}