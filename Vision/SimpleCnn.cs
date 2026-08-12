using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Mathine_test.Vision;

public class SimpleCnn : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor> conv2;

    private readonly Module<Tensor, Tensor> pool;

    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;

    public SimpleCnn()
        : base("SimpleCNN")
    {
        conv1 = Conv2d(
            3,
            16,
            3,
            stride: 1,
            padding: 1);

        conv2 = Conv2d(
            16,
            32,
            3,
            stride: 1,
            padding: 1);

        pool = MaxPool2d(2);

        fc1 = Linear(
            32 * 16 * 16,
            128);

        fc2 = Linear(
            128,
            2);

        RegisterComponents();
    }

    public override Tensor forward(
        Tensor input)
    {
        var x =
            conv1.forward(input);

        x =
            functional.relu(x);

        x =
            pool.forward(x);

        x =
            conv2.forward(x);

        x =
            functional.relu(x);

        x =
            pool.forward(x);

        x =
            x.flatten(1);

        x =
            fc1.forward(x);

        x =
            functional.relu(x);

        x =
            fc2.forward(x);

        return x;
    }
}