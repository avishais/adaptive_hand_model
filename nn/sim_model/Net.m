function x_out = Net(x_in, W, b, x_max, x_min, activation)

l = length(x_in);

T = normzG(x_in, x_max(1:l), x_min(1:l));

for i = 1:numel(W)
    T = T*W{i} + b{i};
    
%     T = tanh(T);
    if i < numel(W)
        
        if activation==1
            T = sigmoid(T);
        else if activation==2
                T = poslin(T);
            else if activation==3
                    T = tanh(T);
                end
            end
        end
        
    end
end

x_out = denormzG(T, x_max(l+1:end), x_min(l+1:end));

end

function x = normz(x, x_max, x_min)

x = (x-x_min)./(x_max-x_min);

end

function x = denormz(x, x_max, x_min)

x = x.*(x_max-x_min) + x_min;

end

function x = normzG(x, mu, sigma)

x = (x-mu)./sigma;
end

function x = denormzG(x, mu, sigma)

x = x.*sigma + mu;

end

function y = sigmoid(x) 
 a = 1;
 c = 0;
 y = 1./(1 + exp(-a.*(x-c)));
end