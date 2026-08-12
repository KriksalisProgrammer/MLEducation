using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class ConceptShiftExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          CONCEPT SHIFT TEST");
        Console.WriteLine("========================================");

        const string trainPath = "Data/concept_train.csv";
        const string normalTestPath = "Data/concept_normal_test.csv";
        const string shiftedTestPath = "Data/concept_shifted_test.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём TRAIN Dataset со СТАРЫМ правилом...");

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
        Console.WriteLine("Создаём CONCEPT SHIFT Dataset...");

        GenerateShiftedDataset(
            shiftedTestPath,
            10_000,
            777);

        Console.WriteLine("CONCEPT SHIFT Dataset создан.");

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

        // TRAIN
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          TRAIN RESULTS");
        Console.WriteLine("========================================");

        var trainMetrics =
            Evaluate(
                mlContext,
                model,
                trainData);

        PrintMetrics(trainMetrics);

        // NORMAL TEST
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       NORMAL TEST RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Тестируем на данных со СТАРЫМ правилом...");

        var normalMetrics =
            Evaluate(
                mlContext,
                model,
                normalTestData);

        PrintMetrics(normalMetrics);

        // CONCEPT SHIFT
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("      CONCEPT SHIFT RESULTS");
        Console.WriteLine("========================================");

        Console.WriteLine(
            "Тестируем на данных с НОВЫМ правилом...");

        Console.WriteLine();
        Console.WriteLine("OLD RULE:");
        Console.WriteLine("Weight > 90 → Overweight");
        Console.WriteLine(
            "Weight > 80 && !IsAthlete → Overweight");

        Console.WriteLine();
        Console.WriteLine("NEW RULE:");
        Console.WriteLine("Weight > 85 → Overweight");
        Console.WriteLine(
            "Weight > 75 && !IsAthlete → Overweight");

        var shiftedMetrics =
            Evaluate(
                mlContext,
                model,
                shiftedTestData);

        PrintMetrics(shiftedMetrics);

        double normalGap =
            trainMetrics.MacroAccuracy -
            normalMetrics.MacroAccuracy;

        double conceptGap =
            trainMetrics.MacroAccuracy -
            shiftedMetrics.MacroAccuracy;

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          CONCEPT SHIFT SUMMARY");
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
            $"Concept Shift Gap:     {conceptGap:P2}");

        Console.WriteLine();

        Console.WriteLine(
            $"Normal LogLoss:        {normalMetrics.LogLoss:F4}");

        Console.WriteLine(
            $"Shifted LogLoss:       {shiftedMetrics.LogLoss:F4}");

        Console.WriteLine();

        Console.WriteLine(
            "SHIFTED CONFUSION MATRIX:");

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

    private static void GenerateShiftedDataset(
        string filePath,
        int count,
        int seed)
    {
        var random =
            new Random(seed);

        using var writer =
            new StreamWriter(filePath);

        for (int i = 0; i < count; i++)
        {
            float height =
                random.Next(150, 201);

            float weight =
                random.Next(45, 121);

            bool isAthlete =
                random.Next(0, 2) == 1;

            float heightMeters =
                height / 100f;

            float bmi =
                weight /
                (heightMeters * heightMeters);

            string result;

            // НОВОЕ ПРАВИЛО
            if (weight > 85)
            {
                result = "Overweight";
            }
            else if (weight > 75 && !isAthlete)
            {
                result = "Overweight";
            }
            else
            {
                result = "Normal";
            }

            writer.Write(
                $"{height.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)}," +
                $"{weight.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)}," +
                $"{isAthlete}," +
                $"{result}," +
                $"{bmi.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)}");

            writer.WriteLine();
        }
    }
}