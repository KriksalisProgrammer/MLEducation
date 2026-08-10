using Microsoft.ML;
using Microsoft.ML.Data;

namespace Mathine_test.Experiments;

public static class RandomFeatureExperiment
{
    public static void Run(MLContext mlContext, string dataPath)
    {
        Console.WriteLine();
        Console.WriteLine("=================================");
        Console.WriteLine("   RANDOM FEATURE EXPERIMENT");
        Console.WriteLine("=================================");

        IDataView data = mlContext.Data.LoadFromTextFile(
            dataPath,
            columns: new[]
            {
                new TextLoader.Column("Height", DataKind.Single, 0),
                new TextLoader.Column("Weight", DataKind.Single, 1),
                new TextLoader.Column("IsAthlete", DataKind.Boolean, 2),
                new TextLoader.Column("Result", DataKind.String, 3),
                new TextLoader.Column("BMI", DataKind.Single, 4),

                // Первые 100 колонок RandomFeature.
                new TextLoader.Column(
                    "RandomFeatures",
                    DataKind.Single,
                    5,
                    104)
            },
            hasHeader: false,
            separatorChar: ',');

        Console.WriteLine("Dataset загружен.");

        Console.WriteLine();
        Console.WriteLine("Колонки Dataset:");

        foreach (var column in data.Schema)
        {
            Console.WriteLine(
                $"{column.Index}: {column.Name} ({column.Type})");
        }

        var randomFeatureColumn =
            data.Schema["RandomFeatures"];

        Console.WriteLine();
        Console.WriteLine(
            $"RandomFeatures type: {randomFeatureColumn.Type}");

        int[] featureCounts = { 0, 5, 10, 20, 50, 100 };

        foreach (int count in featureCounts)
        {
            RunExperiment(
                mlContext,
                data,
                count);
        }
    }

   private static void RunExperiment(
    MLContext mlContext,
    IDataView data,
    int randomFeatureCount)
{
    Console.WriteLine();
    Console.WriteLine(
        $"========== {randomFeatureCount} RANDOM FEATURES ==========");

    var split = mlContext.Data.TrainTestSplit(
        data,
        testFraction: 0.2f,
        seed: 42);

    var trainData = split.TrainSet;
    var testData = split.TestSet;

    Console.WriteLine("Обучение...");

    IEstimator<ITransformer> pipeline =
        mlContext.Transforms.Conversion
            .ConvertType(
                "IsAthleteFloat",
                "IsAthlete");

    pipeline = pipeline.Append(
        mlContext.Transforms.Concatenate(
            "BaseFeatures",
            "Weight",
            "IsAthleteFloat"));

    if (randomFeatureCount > 0)
    {
        pipeline = pipeline.Append(
            mlContext.Transforms.Concatenate(
                "Features",
                "BaseFeatures",
                "RandomFeatures"));
    }
    else
    {
        pipeline = pipeline.Append(
            mlContext.Transforms.CopyColumns(
                "Features",
                "BaseFeatures"));
    }

    pipeline = pipeline.Append(
        mlContext.Transforms.Conversion
            .MapValueToKey(
                "Label",
                "Result"));

    pipeline = pipeline.Append(
        mlContext.MulticlassClassification
            .Trainers
            .SdcaMaximumEntropy());

    pipeline = pipeline.Append(
        mlContext.Transforms.Conversion
            .MapKeyToValue(
                "PredictedLabel"));

    var model = pipeline.Fit(trainData);

    var trainPredictions =
        model.Transform(trainData);

    var testPredictions =
        model.Transform(testData);

    var trainMetrics =
        mlContext.MulticlassClassification.Evaluate(
            trainPredictions);

    var testMetrics =
        mlContext.MulticlassClassification.Evaluate(
            testPredictions);

    double gap =
        trainMetrics.MacroAccuracy -
        testMetrics.MacroAccuracy;

    Console.WriteLine(
        $"TRAIN: {trainMetrics.MacroAccuracy:P2}");

    Console.WriteLine(
        $"TEST:  {testMetrics.MacroAccuracy:P2}");

    Console.WriteLine(
        $"GAP:   {gap:P2}");

    Console.WriteLine(
        $"LogLoss: {testMetrics.LogLoss:F4}");
}
}