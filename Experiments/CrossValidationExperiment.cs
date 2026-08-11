using Mathine_test.Models;
using Microsoft.ML;

namespace Mathine_test.Experiments;

public static class CrossValidationExperiment
{
    public static void Run(
        MLContext mlContext,
        string dataPath)
    {
        Console.WriteLine();
        Console.WriteLine("=================================");
        Console.WriteLine("       CROSS VALIDATION");
        Console.WriteLine("=================================");

        Console.WriteLine();
        Console.WriteLine("Загружаем Dataset...");

        IDataView data =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine("Dataset загружен.");

        var pipeline =
            mlContext.Transforms.Conversion.ConvertType(
                "IsAthleteFloat",
                nameof(PersonData.isAtlete))
            .Append(
                mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(PersonData.Weight),
                    "IsAthleteFloat",
                    nameof(PersonData.BMI)))
            .Append(
                mlContext.Transforms.Conversion.MapValueToKey(
                    "Label",
                    nameof(PersonData.result)))
            .Append(
                mlContext.MulticlassClassification.Trainers
                    .SdcaMaximumEntropy())
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));

        Console.WriteLine();
        Console.WriteLine("Запускаем 10-Fold Cross Validation...");
        Console.WriteLine();

        var results =
            mlContext.MulticlassClassification.CrossValidate(
                data,
                pipeline,
                numberOfFolds: 10);

        Console.WriteLine("=================================");
        Console.WriteLine("          FOLD RESULTS");
        Console.WriteLine("=================================");

        foreach (var result in results.Select((value, index) => new
                 {
                     value,
                     index
                 }))
        {
            Console.WriteLine(
                $"Fold {result.index + 1,2}: " +
                $"Accuracy: {result.value.Metrics.MacroAccuracy:P2} | " +
                $"LogLoss: {result.value.Metrics.LogLoss:F4}");
        }

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
        Console.WriteLine("=================================");
        Console.WriteLine("        FINAL RESULTS");
        Console.WriteLine("=================================");

        Console.WriteLine(
            $"Average Accuracy: {averageAccuracy:P2}");

        Console.WriteLine(
            $"StdDev Accuracy:  {standardDeviation:P2}");

        Console.WriteLine(
            $"Average LogLoss:  {averageLogLoss:F4}");

        Console.WriteLine();
        Console.WriteLine("Cross Validation завершён.");
    }
}