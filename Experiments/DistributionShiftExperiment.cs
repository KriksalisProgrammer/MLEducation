using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class DistributionShiftExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        DISTRIBUTION SHIFT TEST");
        Console.WriteLine("========================================");

        const string trainPath = "Data/shift_train.csv";
        const string normalTestPath = "Data/shift_normal_test.csv";
        const string shiftedTestPath = "Data/shift_test.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём TRAIN Dataset...");
        
        DataGenerator.Generate(
            trainPath,
            10_000,
            0,
            42,
            45,
            120,
            150,
            200);

        Console.WriteLine("TRAIN Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("Создаём NORMAL TEST Dataset...");

        DataGenerator.Generate(
            normalTestPath,
            10_000,
            0,
            999,
            45,
            120,
            150,
            200);

        Console.WriteLine("NORMAL TEST Dataset создан.");

        Console.WriteLine();
        Console.WriteLine("Создаём SHIFTED TEST Dataset...");

        DataGenerator.Generate(
            shiftedTestPath,
            10_000,
            0,
            777,
            60,
            140,
            160,
            210);

        Console.WriteLine("SHIFTED TEST Dataset создан.");

        IDataView trainData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                trainPath,
                hasHeader: false,
                separatorChar: ',');

        IDataView normalTestData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                normalTestPath,
                hasHeader: false,
                separatorChar: ',');

        IDataView shiftedTestData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                shiftedTestPath,
                hasHeader: false,
                separatorChar: ',');

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

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          TRAIN RESULTS");
        Console.WriteLine("========================================");

        var trainMetrics = Evaluate(
            mlContext,
            model,
            trainData);

        PrintMetrics(trainMetrics);

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       NORMAL TEST RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Проверяем на новом Dataset того же распределения...");

        var normalMetrics = Evaluate(
            mlContext,
            model,
            normalTestData);

        PrintMetrics(normalMetrics);

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("      SHIFTED TEST RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Проверяем на Dataset с изменённым распределением...");

        Console.WriteLine(
            "Weight: 60-140");
        Console.WriteLine(
            "Height: 160-210");

        var shiftedMetrics = Evaluate(
            mlContext,
            model,
            shiftedTestData);

        PrintMetrics(shiftedMetrics);

        double normalGap =
            trainMetrics.MacroAccuracy -
            normalMetrics.MacroAccuracy;

        double shiftedGap =
            trainMetrics.MacroAccuracy -
            shiftedMetrics.MacroAccuracy;

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        DISTRIBUTION SHIFT SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine(
            $"TRAIN Accuracy:        {trainMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"NORMAL TEST Accuracy:  {normalMetrics.MacroAccuracy:P2}");

        Console.WriteLine(
            $"SHIFTED TEST Accuracy: {shiftedMetrics.MacroAccuracy:P2}");

        Console.WriteLine();

        Console.WriteLine(
            $"Normal Gap:            {normalGap:P2}");

        Console.WriteLine(
            $"Shifted Gap:           {shiftedGap:P2}");

        Console.WriteLine();

        Console.WriteLine(
            $"Normal LogLoss:        {normalMetrics.LogLoss:F4}");

        Console.WriteLine(
            $"Shifted LogLoss:       {shiftedMetrics.LogLoss:F4}");

        Console.WriteLine();

        if (shiftedMetrics.MacroAccuracy >= 0.99)
        {
            Console.WriteLine(
                "RESULT: Модель устойчива к этому distribution shift.");
        }
        else if (shiftedMetrics.MacroAccuracy >= 0.95)
        {
            Console.WriteLine(
                "RESULT: Модель немного потеряла качество.");
        }
        else if (shiftedMetrics.MacroAccuracy >= 0.90)
        {
            Console.WriteLine(
                "RESULT: Distribution shift заметно влияет на модель.");
        }
        else
        {
            Console.WriteLine(
                "RESULT: Сильная деградация модели.");
        }

        Console.WriteLine();
        Console.WriteLine("SHIFTED CONFUSION MATRIX:");

        Console.WriteLine(
            shiftedMetrics.ConfusionMatrix
                .GetFormattedConfusionTable());
    }

    private static MulticlassClassificationMetrics Evaluate(
        MLContext mlContext,
        ITransformer model,
        IDataView data)
    {
        var predictions =
            model.Transform(data);

        return mlContext.MulticlassClassification.Evaluate(
            predictions,
            labelColumnName: "Label",
            predictedLabelColumnName: "PredictedLabel");
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