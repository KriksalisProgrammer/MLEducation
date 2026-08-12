using Mathine_test.Models;
using Microsoft.ML;

namespace Mathine_test.Experiments;

public static class NoiseRobustnessExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        NOISE & ROBUSTNESS TEST");
        Console.WriteLine("========================================");

        const string dataPath = "Data/noise_train.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём TRAIN Dataset...");

        DataGenerator.Generate(
            dataPath,
            10_000,
            0,
            42,
            45,
            120,
            150,
            200);

        Console.WriteLine("TRAIN Dataset создан.");

        IDataView trainData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
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

        Console.WriteLine();
        Console.WriteLine("Обучение модели...");

        var model = pipeline.Fit(trainData);

        Console.WriteLine("Обучение завершено.");

        var originalData =
            mlContext.Data
                .CreateEnumerable<PersonData>(
                    trainData,
                    reuseRowObject: false)
                .ToList();

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       ROBUSTNESS EXPERIMENT");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine(
            "Проверяем модель при разном уровне шума Weight.");

        Console.WriteLine();

        Console.WriteLine(
            "Noise\t\tAccuracy\tLogLoss");

        Console.WriteLine(
            "----------------------------------------");

        int[] noiseLevels =
        {
            0,
            1,
            3,
            5,
            10
        };

        foreach (int noise in noiseLevels)
        {
            var noisyData =
                AddNoise(
                    originalData,
                    noise);

            var dataView =
                mlContext.Data
                    .LoadFromEnumerable(noisyData);

            var predictions =
                model.Transform(dataView);

            var metrics =
                mlContext.MulticlassClassification.Evaluate(
                    predictions,
                    labelColumnName: "Label",
                    predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine(
                $"{noise,4} kg\t\t" +
                $"{metrics.MacroAccuracy:P2}\t\t" +
                $"{metrics.LogLoss:F4}");
        }

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          ROBUSTNESS SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine(
            "Чем больше шум во входных данных,");
        Console.WriteLine(
            "тем сильнее потенциально падает качество модели.");

        Console.WriteLine();
        Console.WriteLine(
            "Важно: модель обучалась на чистых данных.");
        Console.WriteLine(
            "Шум добавляется только во время тестирования.");
    }

    private static List<PersonData> AddNoise(
        List<PersonData> source,
        int noiseLevel)
    {
        var random =
            new Random(100 + noiseLevel);

        var result =
            new List<PersonData>(
                source.Count);

        foreach (var person in source)
        {
            float noise = 0;

            if (noiseLevel > 0)
            {
                noise =
                    (float)(
                        random.NextDouble() * 2 - 1
                    ) * noiseLevel;
            }

            result.Add(
                new PersonData
                {
                    Height = person.Height,

                    Weight =
                        person.Weight + noise,

                    isAtlete =
                        person.isAtlete,

                    result =
                        person.result,

                    BMI =
                        person.BMI
                });
        }

        return result;
    }
}