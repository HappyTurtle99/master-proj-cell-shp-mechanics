function ForwardSolver
    % Curve forward solver for input pressure + normal forces
    
    close all;

    % Set parameters and functions
    p = 1; % Pressure 
    sig0 = 0.7; % Tension sigma(0)
    L = 2*pi; % Length (initial guess for bvp solver)

    % Solution domain
    xmesh = linspace(0,1,200);

    %%%%% SETTING NORMAL FORCE PROFILE (COMMENT OUT ONE OF THEM) %%%%%%
    %%% (I) Simple nematic profile
    % tn = @(x) 0.8 * ( cos(2*pi*x).^2 - 0.5 );  %Normal force
    % dtn = @(x) 0.8 * (-(2*pi)*2*cos(2*pi*x).*sin(2*pi*x) ); %Normal force der.

    %%% (II) Random tn profile
    NrModes = 3; % How many Fourier modes should be included
    Nmax = 5; % Maximal mode number to draw from ( Nmax >= NrModes )
    N = randperm(Nmax,NrModes)'; % Set of modes
    Magnc = 0.6 * (rand([1,NrModes])' - 0.5); % Magnitudes of each cosine mode
    Magns = 0.6 * (rand([1,NrModes])' - 0.5); % Magnitudes of each sine mode
    tn = @(x) real(exp(1i*2*pi*x*N')*Magnc) + imag(exp(1i*2*pi*x*N')*Magns);
    dtn = @(x) real( 1i*2*pi * (repmat(N',[length(x),1]).*exp(1i*2*pi*x*N')) * Magnc ) ...
             + imag( 1i*2*pi * (repmat(N',[length(x),1]).*exp(1i*2*pi*x*N')) * Magns );

    %%% Test plots of random profiles
    % plot(xmesh',tn(xmesh')); hold on
    % plot(xmesh',dtn(xmesh'));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % ODE function
    function dydx = bvpfcn(x,y,L)
        %y(1) = psi, y(2) = sigma, y(3) = x, y(4) = y
    
        dydx = zeros(4,1);       
        dydx(1:4) = [ ( L*p + dtn(x) ) / y(2); % psi
                      -tn(x) * ( L*p + dtn(x) ) / y(2); % sigma
                      L * cos(y(1)); % x
                      L * sin(y(1)); % y
                     ];              
    end

    % ODE boundary conditions
    function res = bcfcn(ya,yb,~)
        res = [ ya(1); % psi(0) = 0 (eliminates invariance under curve rotation)
                yb(1) - 2*pi % psi(1) = 2*pi (smoothness? closed curve?)
                ya(2) - sig0; % \sigma(0) = sig0 (free parameter?)
                ya(3); % x(0) = 0 (eliminates invariance under x-translation)
                ya(4); % y(0) = 0 (eliminates invariance under y-translation)
              ];
    end
    
    % Create initial guess solution 
    function g = guess(x) % Unit circle
        g = [ 2 * pi * x;      % Psi
              sig0; % sigma
              cos(2 * pi * x);     %x
              sin(2 * pi * x);     %y
              ];
    end
    
    % Set up initial function and parameter guess
    solinit = bvpinit(xmesh, @guess, L); 
    
    % Set solver options
    options = bvpset('RelTol',1e-5,'Nmax',10000,'Stats','on');
    
    % Solve ODE as optimization problem
    sol = bvp5c(@bvpfcn, @bcfcn, solinit, options);

    % Plot shape
    figure(1);
    set(gcf,'color','w');
    subplot(2,2,1)
    plot(sol.y(3,:),sol.y(4,:),'Linewidth',2); 
    title('Shape');
    hold on
    % Mark start and endpoint for visual sanity check of the found shape
    scatter(sol.y(3,1),sol.y(4,1),80,'Linewidth',2);
    scatter(sol.y(3,end),sol.y(4,end),80,'x','Linewidth',2);
    axis equal
    
    % Suitable axis setting
    min_x = min(sol.y(3,:));
    max_x = max(sol.y(3,:));
    min_y = min(sol.y(4,:));
    max_y = max(sol.y(4,:));            
    axis([min_x-0.1, max_x+0.1, min_y-0.1, max_y+0.1]);

    subplot(2,2,2)    
    plot(sol.x,sol.y(2,:),'Linewidth',2);
    title('Tension');

    subplot(2,2,3)    
    plot(sol.x,sol.y(1,:),'Linewidth',2);
    title('Psi');

    subplot(2,2,4)   
    kappa =  ( ( L*p + dtn(sol.x') ) ./ sol.y(2,:)' ) / L; % Curvature
    plot(sol.x,kappa,'Linewidth',2);
    title('Curvature');

    disp('Length');
    disp(sol.parameters);

end