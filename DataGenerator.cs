using System.Globalization;
namespace Mathine_test;

public  class DataGenerator
{
    public void Generate(string filePath, int count)
    {
        var random = new Random(42);
        float randomFeature = (float)random.NextDouble();
        using var writer = new StreamWriter(filePath);

        for (int i = 0; i < count; i++)
        {
            float height = random.Next(150, 201);
            float weight = random.Next(45, 121);
            bool isAthlete = random.Next(0, 2) == 1;

            string result;
            float heightMeters = height / 100f;
            float bmi = weight / (heightMeters * heightMeters);
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

            writer.WriteLine(
                $"{height.ToString(CultureInfo.InvariantCulture)}," +
                $"{weight.ToString(CultureInfo.InvariantCulture)}," +
                $"{isAthlete}," +
                $"{result}," +
                $"{randomFeature.ToString(CultureInfo.InvariantCulture)}," +
                $"{bmi.ToString(CultureInfo.InvariantCulture)}");
        }
    }
}