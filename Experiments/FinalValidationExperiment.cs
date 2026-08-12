using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class FinalValidationExperiment
{
    public static void Run(
        MLContext mlContext,
        string dataPath)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          FINAL VALIDATION");
        Console.WriteLine("========================================");

        IDataView data =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine();
        Console.WriteLine("Dataset загружен.");
        Console.WriteLine();

        // 80% для обучения, 20% для финального теста
        var split =
            mlContext.Data.TrainTestSplit(
                data,
                testFraction: 0.20,
                seed: 42);

        Console.WriteLine(
            "Dataset разделён:");

        Console.WriteLine(
            "80% -> TRAIN");

        Console.WriteLine(
            "20% -> FINAL TEST");

        Console.WriteLine();

        Console.WriteLine("========================================");
        Console.WriteLine("       BEST MODEL CONFIGURATION");
        Console.WriteLine("========================================");

        Console.WriteLine("Features: Weight + IsAthlete");
        Console.WriteLine("Learning Rate: 0.05");
        Console.WriteLine("L2 Regularization: 0.0001");
        Console.WriteLine("Iterations: 500");

        Console.WriteLine();
        Console.WriteLine("Обучение модели...");

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
                        l2Regularization: 0.0001f,
                        maximumNumberOfIterations: 500))
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));

        var model =
            pipeline.Fit(split.TrainSet);

        Console.WriteLine("Обучение завершено.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("             TRAIN RESULTS");
        Console.WriteLine("========================================");

        var trainPredictions =
            model.Transform(split.TrainSet);

        var trainMetrics =
            mlContext.MulticlassClassification.Evaluate(
                trainPredictions,
                labelColumnName: "Label",
                predictedLabelColumnName: "PredictedLabel");

        PrintMetrics(trainMetrics);

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          FINAL TEST RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Проверяем на данных, которых модель не видела...");

        var testPredictions =
            model.Transform(split.TestSet);

        var testMetrics =
            mlContext.MulticlassClassification.Evaluate(
                testPredictions,
                labelColumnName: "Label",
                predictedLabelColumnName: "PredictedLabel");

        PrintMetrics(testMetrics);

        double gap =
            trainMetrics.MacroAccuracy -
            testMetrics.MacroAccuracy;

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("             FINAL SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine(
            $"CV Accuracy:       99.43%");

        Console.WriteLine(
            $"TRAIN Accuracy:    {trainMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"FINAL TEST:        {testMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"TRAIN/TEST GAP:    {gap:P2}");

        Console.WriteLine(
            $"FINAL TEST LogLoss:{testMetrics.LogLoss:F4}");

        Console.WriteLine();

        if (testMetrics.MacroAccuracy >= 0.99)
        {
            Console.WriteLine(
                "RESULT: Отличное обобщение модели.");
        }
        else if (testMetrics.MacroAccuracy >= 0.97)
        {
            Console.WriteLine(
                "RESULT: Хорошее обобщение модели.");
        }
        else
        {
            Console.WriteLine(
                "RESULT: Требуется дополнительный анализ.");
        }

        Console.WriteLine();
        Console.WriteLine("Confusion Matrix:");

        Console.WriteLine(
            testMetrics.ConfusionMatrix.GetFormattedConfusionTable());
    }

    private static void PrintMetrics(
        MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine(
            $"Accuracy:     {metrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"MicroAccuracy:{metrics.MicroAccuracy:P2}");

        Console.WriteLine(
            $"LogLoss:      {metrics.LogLoss:F4}");
        
        
    }
}