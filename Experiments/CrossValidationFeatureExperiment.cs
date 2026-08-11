using Mathine_test.Models;
using Microsoft.ML;

namespace Mathine_test.Experiments;

public static class CrossValidationFeatureExperiment
{
    public static void Run(
        MLContext mlContext,
        string dataPath)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("   CROSS VALIDATION FEATURE COMPARISON");
        Console.WriteLine("========================================");

        IDataView data =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine();
        Console.WriteLine("Dataset загружен.");
        Console.WriteLine("10-Fold Cross Validation");
        Console.WriteLine();

        TestFeatureSet(
            mlContext,
            data,
            "MODEL 1: Weight",
            new[]
            {
                nameof(PersonData.Weight)
            });

        TestFeatureSet(
            mlContext,
            data,
            "MODEL 2: Weight + IsAthlete",
            new[]
            {
                nameof(PersonData.Weight),
                nameof(PersonData.isAtlete)
            });

        TestFeatureSet(
            mlContext,
            data,
            "MODEL 3: Weight + IsAthlete + BMI",
            new[]
            {
                nameof(PersonData.Weight),
                nameof(PersonData.isAtlete),
                nameof(PersonData.BMI)
            });
    }

    private static void TestFeatureSet(
        MLContext mlContext,
        IDataView data,
        string modelName,
        string[] features)
    {
        Console.WriteLine();
        Console.WriteLine("----------------------------------------");
        Console.WriteLine(modelName);
        Console.WriteLine("----------------------------------------");

        IEstimator<ITransformer> pipeline =
            mlContext.Transforms.Conversion.ConvertType(
                "IsAthleteFloat",
                nameof(PersonData.isAtlete));

        var featureColumns = new List<string>();

        foreach (var feature in features)
        {
            if (feature == nameof(PersonData.isAtlete))
            {
                featureColumns.Add("IsAthleteFloat");
            }
            else
            {
                featureColumns.Add(feature);
            }
        }

        pipeline = pipeline.Append(
            mlContext.Transforms.Concatenate(
                "Features",
                featureColumns.ToArray()));

        pipeline = pipeline.Append(
            mlContext.Transforms.Conversion.MapValueToKey(
                "Label",
                nameof(PersonData.result)));

        pipeline = pipeline.Append(
            mlContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy());

        pipeline = pipeline.Append(
            mlContext.Transforms.Conversion.MapKeyToValue(
                "PredictedLabel"));

        Console.WriteLine("Обучение 10 моделей...");

        var results =
            mlContext.MulticlassClassification.CrossValidate(
                data,
                pipeline,
                numberOfFolds: 10);

        var accuracies = results
            .Select(x => x.Metrics.MacroAccuracy)
            .ToList();

        var logLosses = results
            .Select(x => x.Metrics.LogLoss)
            .ToList();

        double averageAccuracy =
            accuracies.Average();

        double averageLogLoss =
            logLosses.Average();

        double variance =
            accuracies
                .Select(x =>
                    Math.Pow(x - averageAccuracy, 2))
                .Average();

        double standardDeviation =
            Math.Sqrt(variance);

        Console.WriteLine();
        Console.WriteLine(
            $"Average Accuracy: {averageAccuracy:P2}");

        Console.WriteLine(
            $"StdDev Accuracy:  {standardDeviation:P2}");

        Console.WriteLine(
            $"Average LogLoss:  {averageLogLoss:F4}");
    }
}