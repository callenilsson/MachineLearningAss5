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
    labels = labels(:);
    y = (bsxfun(@rdivide, exp(x), sum(exp(x), 1)));
    x_is_xc = zeros(features, batch);
    index = sub2ind(size(x_is_xc), labels', 1:batch);
    x_is_xc(index) = -1;
    dldx = (y + x_is_xc)/batch;
    
    %error('Implement this function');
    
end
