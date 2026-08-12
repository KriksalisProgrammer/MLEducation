using Mathine_test.Vision;
using TorchSharp;
using static TorchSharp.torch;

namespace Mathine_test.Experiments;

public static class ImageClassificationExperiment
{
    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("========================================");
        Console.WriteLine("       IMAGE CLASSIFICATION");
        Console.WriteLine("========================================");

        string projectPath =
            Directory.GetParent(
                    AppContext.BaseDirectory)!
                .Parent!
                .Parent!
                .Parent!
                .FullName;

        string trainPath =
            Path.Combine(
                projectPath,
                "Data",
                "Train");

        string validationPath =
            Path.Combine(
                projectPath,
                "Data",
                "Validation");

        string testPath =
            Path.Combine(
                projectPath,
                "Data",
                "Test");

        var train =
            new ImageDataset(
                trainPath);

        var validation =
            new ImageDataset(
                validationPath);

        var test =
            new ImageDataset(
                testPath);

        Console.WriteLine();

        Console.WriteLine(
            $"Train:      {train.Count}");

        Console.WriteLine(
            $"Validation: {validation.Count}");

        Console.WriteLine(
            $"Test:       {test.Count}");

        Console.WriteLine();

        Console.WriteLine(
            "Создаём CNN...");

        var model =
            new SimpleCnn();

        Console.WriteLine(
            "CNN создана.");

        Console.WriteLine();

        Console.WriteLine(
            "Проверяем первое изображение...");

        var image =
            ImageClassificationTrainer.LoadImage(
                train.ImagePaths[0]);

        Console.WriteLine(
            $"Tensor shape: {string.Join(
                " x ",
                image.shape)}");

        using (image)
        {
        }

        model.Dispose();

        Console.WriteLine();
        Console.WriteLine(
            "Первый IMAGE PIPELINE TEST завершён.");
    }
}