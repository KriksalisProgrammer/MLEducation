using System.Globalization;

namespace Mathine_test;

public class DataGenerator
{
    public static void Generate(string filePath, int count, int randomFeatureCount)
    {
        var random = new Random(42);

        using var writer = new StreamWriter(filePath);

        for (int i = 0; i < count; i++)
        {
            float height = random.Next(150, 201);
            float weight = random.Next(45, 121);
            bool isAthlete = random.Next(0, 2) == 1;

            float heightMeters = height / 100f;
            float bmi = weight / (heightMeters * heightMeters);

            string result;

            if (weight > 90)
            {
                result = "Overweight";
            }
            else if (weight > 80 && !isAthlete)
            {
                result = "Overweight";
            }
            else
            {
                result = "Normal";
            }
            writer.Write(
                $"{height.ToString(CultureInfo.InvariantCulture)}," +
                $"{weight.ToString(CultureInfo.InvariantCulture)}," +
                $"{isAthlete}," +
                $"{result}," +
                $"{bmi.ToString(CultureInfo.InvariantCulture)}");
            for (int j = 0; j < randomFeatureCount; j++)
            {
                float randomFeature = (float)random.NextDouble();
                writer.Write(
                    ","+ randomFeature.ToString(CultureInfo.InvariantCulture));
            }
            writer.WriteLine();
        }
    }
}