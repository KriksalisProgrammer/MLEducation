using Mathine_test;
using Mathine_test.Experiments;
using Mathine_test.Services;
using Mathine_test.Models;
using Microsoft.ML;

const string dataPath = "Data/people.csv";
const string modelPath = "Data/model.zip";
var mlService = new MlService();
Console.WriteLine("1 - Train ");
Console.WriteLine("2 - Predict ");
Console.WriteLine("3 - Compare Features");
Console.WriteLine("4 - Multiple Experiments");
Console.WriteLine("5 - Cross Validation");
Console.WriteLine("6 - Cross Validation Feature Comparison");
Console.WriteLine("7 - Hyperparameter Tuning");
Console.WriteLine("8 - Final Validation");
Console.WriteLine("9 - Generalization Test");
Console.WriteLine("10 - Distribution Shift");
Console.WriteLine("11 - Concept Shift");
Console.WriteLine("12 - Data Drift Detection");
Console.WriteLine("13 - Noise & Robustness");
Console.WriteLine("14 - Noise Augmentation");
Console.WriteLine("15 - Image Classification");
Console.WriteLine();
Console.Write("Your choise: ");
var choice = Console.ReadLine();

if (choice == "1")
{
    TrainModel();
}
else if (choice == "2")
{
    Predict();
}
else if (choice == "3")
{
    CompareFeatures();
}
else if (choice == "4")
{
    RunRandomFeatureExperiment();
}
else if (choice == "5")
{
    CrossValidation();
}
else if(choice== "6")
{
    CrossValidationFeatureComparison();
}
else if (choice == "7")
{
    HyperparameterTuning();
}
else if (choice == "8")
{
    FinalValidation();
}
else if (choice == "9")
{
    GeneralizationTest();
}
else if (choice == "10")
{
    DistributionShift();
}
else if (choice == "11")
{
    ConceptShift();
}
else if (choice == "12")
{
    DataDriftDetection();
}
else if (choice == "13")
{
    NoiseRobustness();
}
else if (choice == "14")
{
    NoiseAugmentation();
}
else if (choice == "15")
{
    ImageClassification();
}
else
{
    Console.WriteLine("Неизвестная команда.");
}
void ImageClassification()
{
    ImageClassificationExperiment.Run();
}
void NoiseAugmentation()
{
    Console.WriteLine();
    Console.WriteLine("=== NOISE AUGMENTATION ===");
    Console.WriteLine();

    NoiseAugmentationExperiment.Run(
        new MLContext(seed: 42));
}
void RunRandomFeatureExperiment()
{
    Console.WriteLine();
    Console.WriteLine("=== RANDOM FEATURE EXPERIMENT ===");
    Console.WriteLine();

    DataGenerator.Generate(
        dataPath,
        10_000,
        100);

    Console.WriteLine("Dataset создан!");

    var mlContext = new MLContext(seed: 42);

    RandomFeatureExperiment.Run(
        mlContext,
        dataPath);
}
void NoiseRobustness()
{
    Console.WriteLine();
    Console.WriteLine("=== NOISE & ROBUSTNESS ===");
    Console.WriteLine();

    NoiseRobustnessExperiment.Run(
        new MLContext(seed: 42));
}
void TrainModel()
{
    Console.WriteLine();
    Console.WriteLine("=== TRAIN MODE ===");
    Console.WriteLine();
    

    DataGenerator.Generate(
        dataPath,
        10_000,100);

    Console.WriteLine("Dataset создан!");

    var model = mlService.Train(
        dataPath,
        out var trainingData);

    mlService.SaveModel(
        model,
        trainingData,
        modelPath);

    Console.WriteLine();
    Console.WriteLine("Обучение полностью завершено.");
}
void HyperparameterTuning()
{
    Console.WriteLine();
    Console.WriteLine("=== HYPERPARAMETER TUNING ===");
    Console.WriteLine();

    if (!File.Exists(dataPath))
    {
        DataGenerator.Generate(
            dataPath,
            10_000,
            100);

        Console.WriteLine("Dataset создан!");
    }

    HyperparameterExperiment.Run(
        new MLContext(),
        dataPath);
}
void CompareFeatures()
{
    Console.WriteLine();
    Console.WriteLine("=== FEATURE COMPARISON ===");

    if (!File.Exists(dataPath))
    {
       

        DataGenerator.Generate(
            dataPath,
            10_000,100);

        Console.WriteLine(
            "Dataset создан!");
    }

    mlService.CompareFeatures(
        dataPath);
}
void CompareFeaturesMultipleTimes()
{
   
    Console.WriteLine();
    Console.WriteLine("=== MULTIPLE EXPERIMENTS ===");

    if (!File.Exists(dataPath))
    {
        DataGenerator.Generate(
            dataPath,
            10_000,100);

        Console.WriteLine("Dataset создан!");
    }

    mlService.CompareFeaturesMultipleTimes(
        dataPath);
}
void Predict()
{
    Console.WriteLine();
    Console.WriteLine("=== PREDICT MODE ===");
    Console.WriteLine();

    if (!File.Exists(modelPath))
    {
        Console.WriteLine(
            "Ошибка: модель не найдена.");

        Console.WriteLine(
            "Сначала выполните TRAIN.");

        return;
    }

    var model = mlService.LoadModel(
        modelPath);

    Console.WriteLine();
    Console.WriteLine("Введите данные человека.");
    Console.WriteLine();

    Console.Write("Рост: ");
    var height = float.Parse(
        Console.ReadLine()!);

    Console.Write("Вес: ");
    var weight = float.Parse(
        Console.ReadLine()!);

    Console.Write("Спортсмен? (true/false): ");
    var isAthlete = bool.Parse(
        Console.ReadLine()!);

    var person = new PersonData
    {
        Height = height,
        Weight = weight,
        isAtlete = isAthlete
    };

    var prediction = mlService.Predict(
        model,
        person);

    Console.WriteLine();
    Console.WriteLine("=== RESULT ===");

    Console.WriteLine(
        $"Рост: {person.Height}");

    Console.WriteLine(
        $"Вес: {person.Weight}");

    Console.WriteLine(
        $"Спортсмен: {person.isAtlete}");

    Console.WriteLine();

    Console.WriteLine(
        $"Prediction: {prediction.PredictedResult}");

    Console.WriteLine();

    Console.WriteLine("Scores:");

    foreach (var score in prediction.Score)
    {
        Console.WriteLine(score);
    }
}
void CrossValidationFeatureComparison()
{
    Console.WriteLine();
    Console.WriteLine("=== CROSS VALIDATION FEATURE COMPARISON ===");
    Console.WriteLine();

    if (!File.Exists(dataPath))
    {
        DataGenerator.Generate(
            dataPath,
            10_000,
            100);

        Console.WriteLine("Dataset создан!");
    }

    CrossValidationFeatureExperiment.Run(
        new MLContext(),
        dataPath);
}
void CrossValidation()
{
    Console.WriteLine();
    Console.WriteLine("=== CROSS VALIDATION ===");
    Console.WriteLine();

    if (!File.Exists(dataPath))
    {
        DataGenerator.Generate(
            dataPath,
            10_000,
            100);

        Console.WriteLine("Dataset создан!");
    }

    CrossValidationExperiment.Run(
        new MLContext(),
        dataPath);
}
void FinalValidation()
{
    Console.WriteLine();
    Console.WriteLine("=== FINAL VALIDATION ===");
    Console.WriteLine();

    if (!File.Exists(dataPath))
    {
        DataGenerator.Generate(
            dataPath,
            10_000,
            100);

        Console.WriteLine("Dataset создан!");
    }

    FinalValidationExperiment.Run(
        new MLContext(seed: 42),
        dataPath);
}
void GeneralizationTest()
{
    Console.WriteLine();
    Console.WriteLine("=== GENERALIZATION TEST ===");
    Console.WriteLine();

    GeneralizationExperiment.Run(
        new MLContext(seed: 42));
}
void DistributionShift()
{
    Console.WriteLine();
    Console.WriteLine("=== DISTRIBUTION SHIFT ===");
    Console.WriteLine();

    DistributionShiftExperiment.Run(
        new MLContext(seed: 42));
}
void ConceptShift()
{
    Console.WriteLine();
    Console.WriteLine("=== CONCEPT SHIFT ===");
    Console.WriteLine();

    ConceptShiftExperiment.Run(
        new MLContext(seed: 42));
}
void DataDriftDetection()
{
    Console.WriteLine();
    Console.WriteLine("=== DATA DRIFT DETECTION ===");
    Console.WriteLine();

    DataDriftExperiment.Run(
        new MLContext(seed: 42));
}