using Mathine_test.Models;
using Microsoft.ML;

namespace Mathine_test.Experiments;

public static class DataDriftExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("           DATA DRIFT TEST");
        Console.WriteLine("========================================");

        const string referencePath = "Data/drift_reference.csv";
        const string normalPath = "Data/drift_normal.csv";
        const string driftedPath = "Data/drifted.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём REFERENCE Dataset...");

        DataGenerator.Generate(
            referencePath,
            10_000,
            0,
            42,
            45,
            120,
            150,
            200);

        Console.WriteLine("REFERENCE Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("Создаём NORMAL Dataset...");

        DataGenerator.Generate(
            normalPath,
            10_000,
            0,
            999,
            45,
            120,
            150,
            200);

        Console.WriteLine("NORMAL Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("Создаём DRIFTED Dataset...");

        DataGenerator.Generate(
            driftedPath,
            10_000,
            0,
            777,
            70,
            140,
            160,
            210);

        Console.WriteLine("DRIFTED Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          LOADING DATA");
        Console.WriteLine("========================================");

        IDataView referenceData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                referencePath,
                hasHeader: false,
                separatorChar: ',');

        IDataView normalData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                normalPath,
                hasHeader: false,
                separatorChar: ',');

        IDataView driftedData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                driftedPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine("Datasets загружены.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       NORMAL DATA ANALYSIS");
        Console.WriteLine("========================================");

        Analyze(
            mlContext,
            referenceData,
            normalData,
            "NORMAL DATA");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       DRIFTED DATA ANALYSIS");
        Console.WriteLine("========================================");

        Analyze(
            mlContext,
            referenceData,
            driftedData,
            "DRIFTED DATA");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("           DRIFT SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine("REFERENCE:");
        PrintStatistics(mlContext, referenceData);

        Console.WriteLine();
        Console.WriteLine("NORMAL:");
        PrintStatistics(mlContext, normalData);

        Console.WriteLine();
        Console.WriteLine("DRIFTED:");
        PrintStatistics(mlContext, driftedData);
    }

    private static void Analyze(
        MLContext mlContext,
        IDataView reference,
        IDataView current,
        string name)
    {
        var referencePeople =
            mlContext.Data
                .CreateEnumerable<PersonData>(
                    reference,
                    reuseRowObject: false)
                .ToList();

        var currentPeople =
            mlContext.Data
                .CreateEnumerable<PersonData>(
                    current,
                    reuseRowObject: false)
                .ToList();

        var referenceWeights =
            referencePeople
                .Select(x => (double)x.Weight)
                .ToArray();

        var currentWeights =
            currentPeople
                .Select(x => (double)x.Weight)
                .ToArray();

        double referenceMean =
            referenceWeights.Average();

        double currentMean =
            currentWeights.Average();

        double referenceStd =
            CalculateStdDev(
                referenceWeights,
                referenceMean);

        double currentStd =
            CalculateStdDev(
                currentWeights,
                currentMean);

        double meanDifference =
            Math.Abs(
                currentMean -
                referenceMean);

        double stdDifference =
            Math.Abs(
                currentStd -
                referenceStd);

        double meanDrift =
            meanDifference /
            Math.Max(
                Math.Abs(referenceMean),
                0.0001);

        double stdDrift =
            stdDifference /
            Math.Max(
                Math.Abs(referenceStd),
                0.0001);

        Console.WriteLine();
        Console.WriteLine(name);

        Console.WriteLine();

        Console.WriteLine(
            $"Reference Weight Mean: {referenceMean:F2}");

        Console.WriteLine(
            $"Current Weight Mean:   {currentMean:F2}");

        Console.WriteLine(
            $"Mean Difference:       {meanDifference:F2}");

        Console.WriteLine(
            $"Mean Drift:            {meanDrift:P2}");

        Console.WriteLine();

        Console.WriteLine(
            $"Reference Weight Std:  {referenceStd:F2}");

        Console.WriteLine(
            $"Current Weight Std:    {currentStd:F2}");

        Console.WriteLine(
            $"Std Difference:        {stdDifference:F2}");

        Console.WriteLine(
            $"Std Drift:             {stdDrift:P2}");

        Console.WriteLine();

        if (meanDrift > 0.10 ||
            stdDrift > 0.10)
        {
            Console.WriteLine(
                "RESULT: DATA DRIFT DETECTED");
        }
        else
        {
            Console.WriteLine(
                "RESULT: NO SIGNIFICANT DRIFT");
        }
    }

    private static double CalculateStdDev(
        double[] values,
        double mean)
    {
        double sum = 0;

        foreach (double value in values)
        {
            double difference =
                value - mean;

            sum +=
                difference * difference;
        }

        return Math.Sqrt(
            sum / values.Length);
    }

    private static void PrintStatistics(
        MLContext mlContext,
        IDataView data)
    {
        var people =
            mlContext.Data
                .CreateEnumerable<PersonData>(
                    data,
                    reuseRowObject: false)
                .ToList();

        var weights =
            people
                .Select(x => (double)x.Weight)
                .ToArray();

        double mean =
            weights.Average();

        Console.WriteLine(
            $"Count: {weights.Length}");

        Console.WriteLine(
            $"Min:   {weights.Min():F2}");

        Console.WriteLine(
            $"Max:   {weights.Max():F2}");

        Console.WriteLine(
            $"Mean:  {mean:F2}");

        Console.WriteLine(
            $"Std:   {CalculateStdDev(
                weights,
                mean):F2}");
    }
}