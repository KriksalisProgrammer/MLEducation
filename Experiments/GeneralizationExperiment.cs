using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class GeneralizationExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        GENERALIZATION TEST");
        Console.WriteLine("========================================");

        const string trainingPath = "Data/generalization_train.csv";
        const string newTestPath = "Data/generalization_new.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём TRAIN Dataset...");
        
        DataGenerator.Generate(
            trainingPath,
            10_000,
            42);

        Console.WriteLine("TRAIN Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("Создаём NEW TEST Dataset...");
        
        DataGenerator.Generate(
            newTestPath,
            10_000,
            999);

        Console.WriteLine("NEW TEST Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("           LOADING DATA");
        Console.WriteLine("========================================");

        IDataView trainData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                trainingPath,
                hasHeader: false,
                separatorChar: ',');

        IDataView newTestData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                newTestPath,
                hasHeader: false,
                separatorChar: ',');

        Console.WriteLine("TRAIN Dataset загружен.");
        Console.WriteLine("NEW TEST Dataset загружен.");

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

        var model = pipeline.Fit(trainData);

        Console.WriteLine("Обучение завершено.");

        // TRAIN
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("             TRAIN RESULTS");
        Console.WriteLine("========================================");

        var trainPredictions =
            model.Transform(trainData);

        var trainMetrics =
            mlContext.MulticlassClassification.Evaluate(
                trainPredictions,
                labelColumnName: "Label",
                predictedLabelColumnName: "PredictedLabel");

        PrintMetrics(trainMetrics);

        // NEW DATASET
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       NEW DATASET TEST RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Проверяем модель на полностью новом Dataset...");

        var newTestPredictions =
            model.Transform(newTestData);

        var newTestMetrics =
            mlContext.MulticlassClassification.Evaluate(
                newTestPredictions,
                labelColumnName: "Label",
                predictedLabelColumnName: "PredictedLabel");

        PrintMetrics(newTestMetrics);

        double generalizationGap =
            trainMetrics.MacroAccuracy -
            newTestMetrics.MacroAccuracy;

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          GENERALIZATION SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine(
            $"TRAIN Accuracy:       {trainMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"NEW DATA Accuracy:    {newTestMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"Generalization Gap:   {generalizationGap:P2}");

        Console.WriteLine(
            $"NEW DATA LogLoss:     {newTestMetrics.LogLoss:F4}");

        Console.WriteLine();

        if (newTestMetrics.MacroAccuracy >= 0.99)
        {
            Console.WriteLine(
                "RESULT: Отличное обобщение.");
        }
        else if (newTestMetrics.MacroAccuracy >= 0.97)
        {
            Console.WriteLine(
                "RESULT: Хорошее обобщение.");
        }
        else if (newTestMetrics.MacroAccuracy >= 0.90)
        {
            Console.WriteLine(
                "RESULT: Обобщение приемлемое.");
        }
        else
        {
            Console.WriteLine(
                "RESULT: Требуется анализ обобщения.");
        }

        Console.WriteLine();
        Console.WriteLine("Confusion Matrix:");

        Console.WriteLine(
            newTestMetrics.ConfusionMatrix
                .GetFormattedConfusionTable());
    }

    private static void PrintMetrics(
        MulticlassClassificationMetrics metrics)
    {
        Console.WriteLine(
            $"Accuracy:      {metrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"MicroAccuracy: {metrics.MicroAccuracy:P2}");

        Console.WriteLine(
            $"LogLoss:       {metrics.LogLoss:F4}");
    }
}