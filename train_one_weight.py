import numpy as np

x = np.array([1.0,2.0,3.0])

y_true = np.array([2.0,4.0,6.0])

w = 0.5
learning_rate = 0.1
epochs = 20
for epoch in range(epochs):
    y_pred = w * x

    loss = np.mean((y_pred - y_true)**2)

    gradient = np.mean(2 * x * (y_pred-y_true))
    #上面这步相当于损失值Loss对权重w求导 正好就是求梯度

    #梯度为正说明w应该减小

    #为负则反之

    w = w - learning_rate * gradient

    print(
	f"Epoch{epoch+1:02d} | "
	f"Weight:{w:.6f} | "
	f"Loss:{loss:.6f}"
)



print()
print("=== Training Complete ===")
print(f"Final learned weight:{w:.6f}")

