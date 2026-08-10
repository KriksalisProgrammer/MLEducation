using Mathine_test;
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
    CompareFeaturesMultipleTimes();
}
else
{
    Console.WriteLine("Неизвестная команда.");
}
void TrainModel()
{
    Console.WriteLine();
    Console.WriteLine("=== TRAIN MODE ===");
    Console.WriteLine();
    

    DataGenerator.Generate(
        dataPath,
        10_000,10 );

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
void CompareFeatures()
{
    Console.WriteLine();
    Console.WriteLine("=== FEATURE COMPARISON ===");

    if (!File.Exists(dataPath))
    {
       

        DataGenerator.Generate(
            dataPath,
            10_000,10);

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
            10_000,10);

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