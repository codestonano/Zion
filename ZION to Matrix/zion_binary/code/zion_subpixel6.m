z1 = [0 0];
i1 = [1 0];
o1 = [0 1];
n1 = [1 1];
z = reshape(z1', 2, 1);
i = reshape(i1', 2, 1);
o = reshape(o1', 2, 1);
n = reshape(n1', 2, 1);
%z

z_expand = z;
i_expand = i;
o_expand = o;
n_expand = n;
%z_expand

for x = 1:99
    z_expand = [z_expand, abs(z+rand(size(z))*0.1-0.05)];
    i_expand = [i_expand, abs(i+rand(size(i))*0.1-0.05)];
    o_expand = [o_expand, abs(o+rand(size(o))*0.1-0.05)];
    n_expand = [n_expand, abs(n+rand(size(n))*0.1-0.05)];
end

input = [z_expand, i_expand, o_expand, n_expand];
X = input';
%X

y = zeros(4,400);
y(1,1:100) = 1;
y(2,101:200) = 1;
y(3,201:300) = 1;
y(4,301:400) = 1;
y = y';


theta = ones(2,4);
temp = theta;
h = softmax(X*temp);
J = -sum(sum(y.*log(h)+(1-y).*log(1-h)))./400

for i1 = 1:2
    temp(1) = -temp(1);
    for i2 = 1:2
        temp(2) = -temp(2);
        for i3 = 1:2
	    temp(3) = -temp(3);
	    for i4 = 1:2
	        temp(4) = -temp(4);
	        for i5 = 1:2
	    	    temp(5) = -temp(5);
		    for i6 = 1:2
	   	        temp(6) = -temp(6);
			for i7 = 1:2
			    temp(7) = -temp(7);
			    for i8 = 1:2
       			        temp(8) = -temp(8);
                         		hh = softmax(X*temp);
     		        		JJ = -sum(sum(y.*log(hh)+(1-y).*log(1-hh)))./400;
		        		if(JJ<J)
		          	J = JJ;
                           	theta = temp;
				end
			    end
 		        end
                     end
		end
	    end
	end
    end
end
theta
p = softmax(X*theta)
[maxx,pp] = max(p')




