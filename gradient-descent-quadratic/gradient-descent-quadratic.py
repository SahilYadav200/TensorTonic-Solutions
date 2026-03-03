def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimizes a 1D quadratic function f(x) = ax^2 + bx + c using gradient descent.
    """
    x = x0
    
    for _ in range(steps):
        # 1. Calculate the derivative (gradient) at the current point x
        # The derivative of f(x) = ax^2 + bx + c is f'(x) = 2ax + b
        gradient = 2 * a * x + b
        
        # 2. Update x by taking a step in the opposite direction of the gradient
        x = x - lr * gradient
        
    return x