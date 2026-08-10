using System.Globalization;

namespace Mathine_test;

public class DataGenerator
{
    public static void Generate(string filePath, int count)
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
            
            float randomFeature = (float)random.NextDouble();
            float randomFeature1 = (float)random.NextDouble();
            float randomFeature2 = (float)random.NextDouble();
            float randomFeature3 = (float)random.NextDouble();
            float randomFeature4 = (float)random.NextDouble();
            float randomFeature5 = (float)random.NextDouble();

            writer.WriteLine(
                $"{height.ToString(CultureInfo.InvariantCulture)}," +
                $"{weight.ToString(CultureInfo.InvariantCulture)}," +
                $"{isAthlete}," +
                $"{result}," +
                $"{randomFeature.ToString(CultureInfo.InvariantCulture)}," +
                $"{bmi.ToString(CultureInfo.InvariantCulture)}," +
                $"{randomFeature1.ToString(CultureInfo.InvariantCulture)}," +
                $"{randomFeature2.ToString(CultureInfo.InvariantCulture)}," +
                $"{randomFeature3.ToString(CultureInfo.InvariantCulture)}," +
                $"{randomFeature4.ToString(CultureInfo.InvariantCulture)}," +
                $"{randomFeature5.ToString(CultureInfo.InvariantCulture)}");
        }
    }
}