using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TorchSharp;
using static TorchSharp.torch;

namespace Mathine_test.Vision;

public static class ImageClassificationTrainer
{
    public const int ImageSize = 64;

    public static Tensor LoadImage(string path)
    {
        using var image =
            Image.Load<Rgb24>(path);

        image.Mutate(
            x => x.Resize(
                ImageSize,
                ImageSize));

        var data =
            new float[
                3 *
                ImageSize *
                ImageSize];

        for (int y = 0; y < ImageSize; y++)
        {
            for (int x = 0; x < ImageSize; x++)
            {
                Rgb24 pixel =
                    image[x, y];

                int index =
                    y * ImageSize + x;

                // Red
                data[index] =
                    pixel.R / 255f;

                // Green
                data[
                        ImageSize *
                        ImageSize +
                        index] =
                    pixel.G / 255f;

                // Blue
                data[
                        2 *
                        ImageSize *
                        ImageSize +
                        index] =
                    pixel.B / 255f;
            }
        }

        return tensor(
                data,
                dtype: ScalarType.Float32)
            .reshape(
                1,
                3,
                ImageSize,
                ImageSize);
    }
}