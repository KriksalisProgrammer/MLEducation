using Mathine_test.Models;
using Microsoft.ML;

namespace Mathine_test.Experiments;

public static class HyperparameterExperiment
{
    public static void Run(
        MLContext mlContext,
        string dataPath)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       HYPERPARAMETER GRID SEARCH");
        Console.WriteLine("========================================");

        IDataView data =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine();
        Console.WriteLine("Dataset загружен.");
        Console.WriteLine("Features: Weight + IsAthlete");
        Console.WriteLine();

        float[] learningRates =
        {
            0.03f,
            0.05f,
            0.07f,
            0.1f,
            0.15f
        };

        float[] l2Values =
        {
            0.0001f,
            0.001f,
            0.005f,
            0.01f
        };

        int[] iterations =
        {
            100,
            200,
            500
        };

        var bestAccuracy = double.MinValue;
        HyperparameterConfig? bestConfig = null;

        int total =
            learningRates.Length *
            l2Values.Length *
            iterations.Length;

        int current = 0;

        Console.WriteLine(
            $"Всего конфигураций: {total}");

        foreach (var learningRate in learningRates)
        {
            foreach (var l2 in l2Values)
            {
                foreach (var iteration in iterations)
                {
                    current++;

                    var config = new HyperparameterConfig(
                        learningRate,
                        l2,
                        iteration);

                    Console.WriteLine();
                    Console.WriteLine(
                        $"[{current}/{total}]");

                    Console.WriteLine(
                        $"LR={learningRate} | " +
                        $"L2={l2} | " +
                        $"Iterations={iteration}");

                    var result =
                        TestConfiguration(
                            mlContext,
                            data,
                            config);

                    Console.WriteLine(
                        $"Accuracy: {result.Accuracy:P2}");

                    Console.WriteLine(
                        $"StdDev:   {result.StdDev:P2}");

                    Console.WriteLine(
                        $"LogLoss:  {result.LogLoss:F4}");

                    if (result.Accuracy > bestAccuracy)
                    {
                        bestAccuracy = result.Accuracy;
                        bestConfig = config;

                        Console.WriteLine(
                            ">>> NEW BEST CONFIGURATION <<<");
                    }
                }
            }
        }

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("           BEST CONFIGURATION");
        Console.WriteLine("========================================");

        if (bestConfig != null)
        {
            Console.WriteLine(
                $"Learning Rate:     {bestConfig.LearningRate}");

            Console.WriteLine(
                $"L2 Regularization: {bestConfig.L2Regularization}");

            Console.WriteLine(
                $"Iterations:        {bestConfig.Iterations}");

            Console.WriteLine(
                $"Accuracy:          {bestAccuracy:P2}");
        }

        Console.WriteLine();
        Console.WriteLine("Grid Search завершён.");
    }

    private static ExperimentResult TestConfiguration(
        MLContext mlContext,
        IDataView data,
        HyperparameterConfig config)
    {
        var pipeline =
            mlContext.Transforms.Conversion.ConvertType(
                "IsAthleteFloat",
                nameof(PersonData.isAtlete))
            .Append(
                mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(PersonData.Weight),
                    "IsAthleteFloat"))
            .Append(
                mlContext.Transforms.Conversion.MapValueToKey(
                    "Label",
                    nameof(PersonData.result)))
            .Append(
                mlContext.MulticlassClassification.Trainers
                    .SdcaMaximumEntropy(
                        labelColumnName: "Label",
                        featureColumnName: "Features",
                        l2Regularization:
                            config.L2Regularization,
                        maximumNumberOfIterations:
                            config.Iterations))
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));

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
                    Math.Pow(
                        x - averageAccuracy,
                        2))
                .Average();

        double standardDeviation =
            Math.Sqrt(variance);

        return new ExperimentResult(
            averageAccuracy,
            standardDeviation,
            averageLogLoss);
    }

    private record HyperparameterConfig(
        float LearningRate,
        float L2Regularization,
        int Iterations);

    private record ExperimentResult(
        double Accuracy,
        double StdDev,
        double LogLoss);
}