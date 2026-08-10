using Microsoft.ML.Data;
namespace Mathine_test.Models;

public class PersonData
{
    
    [LoadColumn(0)]
    public float Height { get; set; }
    [LoadColumn(1)]
    public float Weight { get; set; }
    [LoadColumn(2)] 
    public bool isAtlete { get; set; }

    [LoadColumn(3)] 
    public string result { get; set; } = "";
    [LoadColumn(4)]
    public float RandomFeature { get; set; }
    [LoadColumn(5)]
    public float BMI { get; set; }
    [LoadColumn(6)]
    public float RandomFeature1 { get; set; }

    [LoadColumn(7)]
    public float RandomFeature2 { get; set; }

    [LoadColumn(8)]
    public float RandomFeature3 { get; set; }

    [LoadColumn(9)]
    public float RandomFeature4 { get; set; }

    [LoadColumn(10)]
    public float RandomFeature5 { get; set; }

}

public class PersonPrediction
{
    [ColumnName("PredictedLabel")] public string PredictedResult { get; set; } = "";
    public float[] Score { get; set; } = [];
}