function test_relu()
    addpath(genpath('../'));
    
    x = [2 4 -5 6 7 -1;
        -4 0 0 3 4 -5];
    res = [2 4 0 6 7 0;
        0 0 0 3 4 0];
    test_equal(relu_forward(x), res, ...
        1e-5, 'Forward pass with batch of one is incorrect');

    % gradients
    for k=1:5
        x = randn(21, 34, 4);
        y = relu_forward(x);
        dldx = relu_backward(x, y);
        test_gradients(@(x) 0.5*sum(vec(relu_forward(x).^2)), dldx, x, 1e-5, 30);
    end

    fprintf('Everything passed! Your implementation seems correct.\n');
end