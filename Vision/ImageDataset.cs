using TorchSharp;
using static TorchSharp.torch;

namespace Mathine_test.Vision;

public class ImageDataset
{
    public List<string> ImagePaths { get; }
    public List<long> Labels { get; }

    public int Count => ImagePaths.Count;

    public ImageDataset(string directory)
    {
        ImagePaths = new List<string>();
        Labels = new List<long>();

        LoadClass(
            Path.Combine(directory, "cats"),
            0);

        LoadClass(
            Path.Combine(directory, "dogs"),
            1);
    }

    private void LoadClass(
        string directory,
        long label)
    {
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException(directory);

        var files = Directory
            .GetFiles(directory)
            .Where(IsImage)
            .ToList();

        foreach (var file in files)
        {
            ImagePaths.Add(file);
            Labels.Add(label);
        }
    }

    private static bool IsImage(string path)
    {
        string extension =
            Path.GetExtension(path)
                .ToLowerInvariant();

        return extension is
            ".jpg" or
            ".jpeg" or
            ".png" or
            ".bmp";
    }
}