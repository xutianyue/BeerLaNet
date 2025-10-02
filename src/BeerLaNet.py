import torch
import torch.nn as nn
import torch.nn.functional as F

class BeerLaNet(nn.Module):
    
    def __init__(self, r, c=3, learn_S_init = False, calc_tau = True):
        """
        A layer to perform color normalization based on a sparse, low-rank non-negative
        matrix factorization model of the form:
            
        min_{x_0,S,D} 1/2|X + S*D^T - x_0*1^T|_F^2 + \lambda \sum_{i=1}^r |s_i|_2 (\gamma |d_i|_1 + |d_i|_2)
        s.t. S>=0, D>=0
        
        This is accomplished through alternating updates through each block of
        variables (x_0,S,D).
        
        In the above equation X is assumed to be reshaped to have dimension
        c x p where c is the number of color channels in the image (typically 3
        for RGB images) and p is the number of pixels in the image.  For the
        actual forward operator of the layer, however, X will be input in typical 
        pytorch format of [n,p1,p2,c] where n is the number of images in the mini-batch
        (p1,p2) are the image dimenions, and c is the number of color channels.
        
        In general, the sequence of updates proceeds in order x_0 -> D -> S.
        The initialization for x_0 and D is always taken to be zeros if it is 
        not passed as an input to forward(), while the initialization for S is
        a learned parameter if not passed as an input to forward.
        
        Parameters
        ----------
        r : int
            The number of columns in S and D to use for the layer
            
        c : int
            The first dimension of S (the color spectrum matrix).  
            
        learn_S_init : boolean
            If true then the initialization for S will be a learnable parameter.
            If false, then an initialization for S must be passed as an input
            to the forward function.
            
        calc_tau : boolean
            If true, then the step size parameter (tau) will be calculated based
            on a norm of S/D.  If false, then this parameter will be trainable.
            
        """
        
        super(BeerLaNet, self).__init__()
        self.r = r
        self.c = c
        
        self.calc_tau = calc_tau
        
        self.gamma = nn.Parameter(torch.rand(1)*1e-5)
        self.lam   = nn.Parameter(torch.rand(1)*1e-5)
        
        if not calc_tau:
            self.tau   = nn.Parameter(torch.rand(1)*1e-5)
        
        if learn_S_init:
            #initialize S with uniform variables to be non-negative
            self.S_init = nn.Parameter(torch.rand(self.c,self.r))
            
            with torch.no_grad():
                self.S_init.data = self.S_init.data/self._S_norm(self.S_init.data)
                
        else:
            self.S_init = None        
        
    def forward(self, X, S=None, D=None, n_iter=1, unit_norm_S=True):
        """
        This performs update iterations for the matrix factorization color
        normalization model described in the constructor.  It does this by a 
        sequence of updates to the object in the order x_0 -> D -> S, where
        the update for x_0 is a closed form optimal update, and the updates for
        S and D are via proximal gradient descent updates.

        Parameters
        ----------
        X : pytorch tensor [n,c,p1,p2]
            The input data tensor, where n is the number of minibatch samples,
            (p1,p2) are the image dimensions, and c is the number of color
            channels (or feature channels more generally).  c should be equal
            to the value of c passed to the constructor
            
        S : pytorch tensor [c,r]
            The initial input spectra for the layer.  If this is not provided
            then the layer must have been constructed with learn_S_init=True
        
        D : pytorch tensor [n,r,p1,p2], optional
            The initial input for the density maps.  If this is not provided
            then this will be initialized as all zeros.
        
        n_iter : int, optional
            The number of iterations of the optimization to run in this layer.
            The default is 1.
        
        unit_norm_S : boolean, optional
            If true, then S and D will be rescaled after each iteration so that 
            S has unit norm columns.  Note this does not effect the objective value.

        Returns
        -------
        x_0 : pytorch tensor [c]
            The estimate of the background intensity.
            
        S : pytorch tensor [c,r]
            The current spectrum matrix
            
        D : pytorch tensor [n,r,p1,p2]
            The current optical density maps.

        """

        n,c_in,p1,p2 = X.shape
        p = p1*p2
        
        assert c_in == self.c
        
        #Resahpe the X data to be in matrix form with size [n,c,p]
        X = X.view(-1,self.c,p1*p2)
        
        if S is None:
            S = self.S_init.clone().to(X.device)
        
        if D is None:
            Dt = torch.zeros(n,self.r,p, device=X.device) #This is D^T 
        else:
            Dt = D.view(-1,self.r,p) #This is D^T 
            
        #Make sure the regularization and step size parameters are non-negative
        #Since these are learnable parameters the optimizer can push these
        #negative, so we take the absolute value to prevent this.
        with torch.no_grad():
            self.gamma.data = torch.abs(self.gamma.data)
            self.lam.data   = torch.abs(self.lam.data)
            
            if not self.calc_tau:
                self.tau.data   = torch.abs(self.tau.data)
        
        #Now run the main computation
        
        for _ in range(n_iter):
            SDt = S@Dt # compute S*D^T [n,c,p]
            
            ######################
            #Compute x_0 [n,c,1]
            #We keep the final dimension for broadcasting later
            x_0 = torch.mean(X+SDt,dim=2,keepdims=True)
            
            ######################
            #Now start the updates for D.
            #Here we'll be make the updates with D shaped as D^T
            
            #First the gradient step for Dt
            #Dt = Dt - tau*(S^T*S*D^T + S^T*X - S^T*x_0*1^T)
            
            if self.calc_tau:
                tau_D = 1.0/torch.linalg.matrix_norm(S, ord='fro')**2
            else:
                tau_D = self.tau
            
            Dt += -tau_D*(S.T@SDt + S.T@X - S.T@x_0)
            
            # #Now compute the proximal operator.  This is the composition
            # #of first doing soft-thresholding, followed by scaling
            
            # #First compute the soft-thresholding for the L1 proximal operator
            S_nrm = self._S_norm(S).view(1,self.r,1)
            Dt = F.relu(Dt-self.lam*self.gamma*tau_D*S_nrm)
            
            # #Now compute the scaling for the L2 proximal operator
            Dt_L2 = self._Dt_Lp(Dt,2)
            scl = F.relu(Dt_L2-self.lam*tau_D*S_nrm)
            scl = scl/Dt_L2+1e-10
            Dt = Dt*scl
        
            # ######################
            # #Now the updates for S.
            
            # #First update SDt
            SDt = S@Dt
            
            # #Also update x_0
            x_0 = torch.mean(X+SDt,dim=2,keepdims=True)
            
            # #Now the gradient step for S
            Dt_sum = Dt.sum(dim=2,keepdim=True)
            
            #The gradient step for a single image is given as
            #S = S - tau*(S*D^T*D + X*D - x_0*1^T*D)
            #
            #but here we can have multiple images in a batch, so we take the
            #mean over the batch dimension
            
            if self.calc_tau:
                tau_S = 1.0/torch.mean(torch.linalg.matrix_norm(Dt, ord='fro')**2)
            else:
                tau_S = self.tau
            
            #We rename the variable to avoid inline modification errors for backprop
            S_grad = S-tau_S*torch.mean(SDt@Dt.permute(0,2,1) + X@Dt.permute(0,2,1) - x_0@Dt_sum.permute(0,2,1),
                             dim=0, keepdims=False)
            
            #Now compute the proximal operator for the L2 norm
            #Here we compute the mean norms for Dt across the batch.
            Dt_nrm = self._Dt_norm(Dt).mean(dim=0,keepdims=False) #[r,1]
            S_nrm = self._S_norm(S_grad) #[1,r]
            scl_S = F.relu(S_nrm-self.lam*tau_S*Dt_nrm.T) #[1,r]
            scl_S = scl_S/(S_nrm+1e-10)
            
            S = S_grad*scl_S
            
            # ######################
            # #All the updates are done.
            # #We can rescale to have S with unit norm if desired.
            
            if unit_norm_S:
                 S_nrm = self._S_norm(S)
                 S = S/(S_nrm+1e-10)
                 Dt = Dt*(S_nrm.view(1,self.r,1)+1e-10)
                
                
        return x_0, S, Dt.view(n,self.r,p1,p2)
            
            
    def _S_norm(self,S):
        """
        Returns the L2 norms of S

        Parameters
        ----------
        S : pytorch tensor [c,r]

        Returns
        -------
        norms_S : pytorch tensor [1,r]
        
            norms_S[0,i] corresponds to the L2 norm of the i'th row of S

        """
        
        return torch.linalg.vector_norm(S,ord=2,dim=0,keepdim=True)
    
    def _Dt_Lp(self,Dt,nrm_ord):
        """
        Returns the Lp norm for the columns of D (or the rows of D^T)

        Parameters
        ----------
        Dt : pytorch tensor [n,r,p]
            The tensor containing D^T

        nrm_ord : scaler
            The ord parameter of the norm (see torch.linalg.vector_norm)

        Returns
        -------
        norm_Dt : pytorch tensor [n,r,1]
            norm_L2[i,j,0] corresponds to the Lp norm for the j^th row of the
            optical density map for the i^th image.

        """
        
        return torch.linalg.vector_norm(Dt,ord=nrm_ord,dim=2,keepdim=True)
    
    def _Dt_norm(self,Dt):
        """
        Returns the norm self.gamma||D_i||_1 + ||D_i||_2 for the columns of
        D (or the rows of D^T)

        Parameters
        ----------
        Dt : pytorch tensor [n,r,p]
            The tensor containing D^T

        Returns
        -------
        norms_Dt : pytorch tensor [n,r,1]
            norms_Dt[i,j,0] corresponds to the norm for the j^th row of the
            optical density map for the i^th image.

        """
        
        return self.gamma*self._Dt_Lp(Dt,1) + self._Dt_Lp(Dt,2)

