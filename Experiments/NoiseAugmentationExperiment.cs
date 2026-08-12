using Mathine_test.Models;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class NoiseAugmentationExperiment
{
    public static void Run(MLContext mlContext)
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        NOISE AUGMENTATION TEST");
        Console.WriteLine("========================================");

        const string dataPath =
            "Data/augmentation_train.csv";

        Console.WriteLine();
        Console.WriteLine("Создаём базовый Dataset...");

        DataGenerator.Generate(
            dataPath,
            10_000,
            0,
            42,
            45,
            120,
            150,
            200);

        IDataView cleanData =
            mlContext.Data.LoadFromTextFile<PersonData>(
                dataPath,
                hasHeader: false,
                separatorChar: ',');

        var cleanPeople =
            mlContext.Data
                .CreateEnumerable<PersonData>(
                    cleanData,
                    reuseRowObject: false)
                .ToList();

        Console.WriteLine(
            $"Clean samples: {cleanPeople.Count}");

        Console.WriteLine();
        Console.WriteLine("Создаём augmentation...");

        var augmentedPeople =
            CreateAugmentedDataset(
                cleanPeople);

        Console.WriteLine(
            $"Augmented samples: {augmentedPeople.Count}");

        IDataView augmentedData =
            mlContext.Data.LoadFromEnumerable(
                augmentedPeople);

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       TRAIN MODEL A");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine(
            "MODEL A: обучение только на чистых данных.");

        var cleanModel =
            TrainModel(
                mlContext,
                cleanData);

        Console.WriteLine();
        Console.WriteLine("MODEL A готов.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       TRAIN MODEL B");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine(
            "MODEL B: обучение на clean + noise.");

        var augmentedModel =
            TrainModel(
                mlContext,
                augmentedData);

        Console.WriteLine();
        Console.WriteLine("MODEL B готов.");

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("        ROBUSTNESS COMPARISON");
        Console.WriteLine("========================================");

        int[] noiseLevels =
        {
            0,
            1,
            3,
            5,
            10
        };

        Console.WriteLine();
        Console.WriteLine(
            "Noise\tMODEL A\tMODEL B");

        Console.WriteLine(
            "----------------------------------------");

        foreach (int noiseLevel in noiseLevels)
        {
            var testPeople =
                AddNoise(
                    cleanPeople,
                    noiseLevel);

            IDataView testData =
                mlContext.Data
                    .LoadFromEnumerable(
                        testPeople);

            var metricsA =
                EvaluateModel(
                    mlContext,
                    cleanModel,
                    testData);

            var metricsB =
                EvaluateModel(
                    mlContext,
                    augmentedModel,
                    testData);

            Console.WriteLine(
                $"{noiseLevel,2} kg\t" +
                $"{metricsA.MacroAccuracy:P2}\t" +
                $"{metricsB.MacroAccuracy:P2}");
        }

        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("          AUGMENTATION SUMMARY");
        Console.WriteLine("========================================");

        Console.WriteLine();
        Console.WriteLine(
            "MODEL A = обучение без шума.");

        Console.WriteLine(
            "MODEL B = обучение с шумом.");

        Console.WriteLine();
        Console.WriteLine(
            "Сравниваем устойчивость обеих моделей");
        Console.WriteLine(
            "на одинаковых зашумлённых данных.");
    }

    private static ITransformer TrainModel(
        MLContext mlContext,
        IDataView data)
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
                        l2Regularization: 0.0001f,
                        maximumNumberOfIterations: 500))
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(
                    "PredictedLabel"));

        return pipeline.Fit(data);
    }

    private static MulticlassClassificationMetrics EvaluateModel(
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

    private static List<PersonData> CreateAugmentedDataset(
        List<PersonData> source)
    {
        var result =
            new List<PersonData>();

        result.AddRange(source);

        int[] noiseLevels =
        {
            1,
            3,
            5,
            10
        };

        foreach (int noiseLevel in noiseLevels)
        {
            result.AddRange(
                AddNoise(
                    source,
                    noiseLevel));
        }

        return result;
    }

    private static List<PersonData> AddNoise(
        List<PersonData> source,
        int noiseLevel)
    {
        var random =
            new Random(
                1000 + noiseLevel);

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