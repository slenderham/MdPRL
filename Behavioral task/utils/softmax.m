function [p] = softmax(logits)
    assert(size(logits, 2)==1)
    p = exp(logits)./sum(exp(logits),1);