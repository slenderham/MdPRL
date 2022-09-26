function [p] = softmax(logits)
    assert(size(logits, 2)==1)
    logits = logits-max(logits, [], "all");
    p = exp(logits)./sum(exp(logits),1);
end