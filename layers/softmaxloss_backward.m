function dldx = softmaxloss_backward(x, labels)
    % Inputs:
    %    x - Features. See the reshape command below. It is reshaped as for
    %        the fully connected layer.
    %    y - Labels. It is a vector with the correct labels. For
    %        instance if we have a batch of two where the first example is
    %        class 4 and the second example is class 7, labels is [4 7].
    %
    % Outputs:
    %    dldx - Partial derivative of L with respect to x. Remember that in
    %           the forward pass you average over the batch elements.
    sz = size(x);
    batch = sz(end);
    features = prod(sz(1:end-1));

    % suitable for matrix multiplication
    %x = reshape(x, [features, batch]);
    % for numerical stability. Convince yourself that the result is the same.
    %x = bsxfun(@minus, x, min(x, [], 1));
    
    %MIN KOD HÄR
    
    %x = bsxfun(@minus, x, min(x, [], 1));
    x;
    dldx = zeros(sz);
    labels;
    
    sum2 = sum(exp(x));
    
    for j = 1:sz(2)
        for i = 1:sz(1)
            if i ==  labels(j)
                dldx(i,j) = dldx(i,j) - 1;
            end
            dldx(i,j) = dldx(i,j) + exp(x(i,j))/sum2(j);
        end
    end
    
    dldx
    
    %error('Implement this function');
    
end
