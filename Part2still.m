
xy = [1.0, 1.0];

  descend('r', 0);
  descend('b', 1);
  show()

function xyRtheta = xy_for_rtheta(rtheta)
    r = rtheta(0);
    theta = rtheta(1);
    xyRtheta = [2*r*np.cos(theta), 2*r*np.sin(theta)];
end

function dx_dy = err(rtheta)
    x, y = xy_for_rtheta(rtheta);
    dx = x-xy(0);
    dy = y-xy(1);
    dx_dy = dx*dx + dy*dy;
end

function grad1= compute_grad(rtheta, natural = True)
    grad = gradient(rtheta, err, 1E-6); %epsilon = 1E-6
end


function grad1= compute_Ngrad(rtheta, natural = True)
    grad = gradient(rtheta, err, 1E-6); %epsilon = 1E-6

      G = [[1.0, 0.0], [0.0, rtheta(0)^2]];
      grad = inv(G)* grad;
      grad1 =  grad / norm(grad);
end


function gd = descend(color, natural)

    rtheta = [0.5, np.pi*3/4.];
    rthetas = rtheta;
    stepsize = 0.001;
    tol = 0.0001;

    while err(rtheta) > tol
        rtheta = rtheta - stepsize * compute_grad(rtheta, natural);
        rthetas.append(rtheta);
        rthetas = xy_for_rtheta(rthetas);
        
    xs, ys = zip(rthetas);
    gd =  plot(xs, ys, color);
    end
    
end
    