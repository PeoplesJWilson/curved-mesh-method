from scipy.spatial import KDTree, Delaunay
from scipy.sparse import csr_matrix

from numpy import zeros, ones, diag, dot, trace, transpose, outer, stack, array, unique, sqrt
from numpy.linalg import eigh, det, pinv, lstsq

from sklearn.preprocessing import PolynomialFeatures


class CurvedMeshMethod:

    def __init__(self, n_neighbors, dim = 2, stiffness_diagonalization = None, grid_size = 3):
        # user input 
        self.n_neighbors = n_neighbors
        self.dim = dim
        self._stiffness_diagonalization = stiffness_diagonalization
        self._grid_size = grid_size 
        
        
        # for storing local chart data
        self._tangent_basis = {}
        self._coordinates = None
        self._mls_target = None
        self._knn_indices = None
        self._local_mesh = None
        self._coefs = None

        # for storing matrix constuction
        self._function_mass_matrix = None
        self._function_stiffness_matrix = None
        self._vector_field_stiffness_matrix = None
        self._vector_field_mass_matrix = None
    
        # error handling 
        self.__allowed_tensors = ['function', 'vector-field']
        self.__diagonalization_strategies = [None, "uniform-data", "random-data"]
#______________________________________________________
#______________________________________________________
#______________________________________________________
# operator specific matrix constructions
        
    def _new_custom_operator(self, data, func, tensor='vector-field'):
        if tensor == 'function':
            if self._function_mass_matrix is None:
                self._compute_function_mass_matrix(data)
            
            return self._compute_function_stiffness_matrix(data, func), self._function_mass_matrix
        
        elif tensor == 'vector-field':
            if self._vector_field_mass_matrix is None:
                self._compute_vector_field_mass_matrix(data)
            
            return self._compute_function_stiffness_matrix(data, func), self._vector_field_mass_matrix

        else:
            raise KeyError(f"{tensor} is not in {self.__allowed_tensors}")

    def _weak_laplace_beltrami(self, data):
            
            # implement the integral 
        def _laplace_beltrami_stiffness_integrand(polynomial, points, u_1, u_2):
            g = self._riemannian_metric(polynomial, points, u_1, u_2)
            det_g = det(g)
            integrand = (-g[:,1,1]/(det_g) + g[:,0,1]/(det_g))*sqrt(det_g)
            return integrand.mean()
                        
        self._function_stiffness_matrix = self._compute_function_stiffness_matrix(data, _laplace_beltrami_stiffness_integrand)

        if self._function_mass_matrix is None:
            self._compute_function_mass_matrix(data)
        
    def _weak_bochner_laplacian(self, data):
        
        self._vector_field_stiffness_matrix = self._compute_vector_field_stiffness_matrix(data, self._bochner_stiffness_integrand)
 
        if self._vector_field_mass_matrix is None:
            self._compute_vector_field_mass_matrix(data)

    def _weak_hodge_laplacian(self, data):
        self._vector_field_stiffness_matrix = self._compute_vector_field_stiffness_matrix(data, self._hodge_stiffness_integrand)
 
        if self._vector_field_mass_matrix is None:
            self._compute_vector_field_mass_matrix(data)     
#______________________________________________________
#______________________________________________________
#______________________________________________________
# operator space specific constuctions (function, vectorfields)
        
    def _compute_function_mass_matrix(self, data):

        # resolve dependencies for this method
        if self._local_mesh is None:
            self._compute_local_mesh(data)
        
        if self._coefs is None:
            self._compute_mls_coefs(data)

        u_1, u_2 = self._get_triangle_grid()
        N = data.shape[0]
 
        mass_rows, mass_cols, mass_data = [],[],[]   # store data for csr_matrix

        # construct rows of mass matrix
        for index in range(N):

            polynomial = self._coefs[index]                 # row-wise constants
            neighborhood = self._coordinates[index]
            [_, first_ring, vertices] = self._local_mesh[index]    

            acc = 0     # for diagonal element, sum each integral around triangle in first ring
            for triangle in first_ring:
                triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 1)    
                points = neighborhood[triangle]     # [base_point, vertex, other]

                g, hat = self._riemannian_metric(polynomial, points, u_1, u_2), (1 - u_1 - u_2)
                integrand = sqrt(det(g))*(hat**2)   # integrate over triangle
                acc += integrand.mean()
    
   
            mass_rows.append(index)         # data for csr_matrix
            mass_cols.append(index)
            mass_data.append(acc)
            
   
            for vertex in vertices:     # off diagonals: sum along triangles with edge p_index <---> p_jndex
                jndex = self._knn_indices[index, vertex]    
                triangle_pair = first_ring[(first_ring == vertex).any(axis = 1),:]

                acc = 0
                for triangle in triangle_pair:
                    triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 0 if x == vertex else 1)
                    points = neighborhood[triangle, :]  # [basepoint, vertex, other]

                    g, hat =  self._riemannian_metric(polynomial, points, u_1, u_2), (1 - u_1 - u_2)
                    integrand = sqrt(det(g))*(hat)*u_1  # integrate over triangle
                    acc += integrand.mean()
    
                mass_rows.append(index)       # data for csr_matrix
                mass_cols.append(jndex)
                mass_data.append(acc)
            
        mass = csr_matrix((mass_data, (mass_rows, mass_cols)), shape=(N, N))      
        self._function_mass_matrix = .5*(mass + mass.T) 

    def _compute_vector_field_mass_matrix(self, data):
        dim = self.dim
        N = data.shape[0]
        u_1, u_2 = self._get_triangle_grid()          # constants


        mass_rows, mass_cols, mass_data = [],[],[]
        # through each block of (dim) rows 

        for index in range(N):
            neighborhood, polynomial = self._coordinates[index], self._coefs[index]
            Basis_i, _ = self._compute_nodal_basis(index, index)

            [_, first_ring, vertices] = self._local_mesh[index]

            # first, compute diagonal block
            for a_ind in range(dim):
                for b_ind in range(dim):
    
                    acc = 0
                    for triangle in first_ring:
                        triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 1)
                        points = neighborhood[triangle, :]
                        acc += self._vector_field_mass_integrand(polynomial, points, 0, Basis_i[a_ind], Basis_i[b_ind], u_1, u_2).mean()
                
                    mass_rows.append(index*dim + a_ind)
                    mass_cols.append(index*dim + b_ind)
                    mass_data.append(acc)


            # now, loop through off diagonal triangle pairs
            for vertex in vertices:
                jndex = self._knn_indices[index, vertex]
                Basis_i, Basis_j = self._compute_nodal_basis(index, jndex)

                triangle_pair = first_ring[(first_ring == vertex).any(axis = 1),:]  # identify pair of triangles with x_i x_j as a side 

                for a_ind in range(dim):
                    for b_ind in range(dim):

                        acc = 0
                        for triangle in triangle_pair:
                            triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 0 if x == vertex else 1) 
                            points = neighborhood[triangle, :]
                            acc += self._vector_field_mass_integrand(polynomial, points, vertex, Basis_i[a_ind], Basis_j[b_ind], u_1, u_2).mean()

                        mass_rows.append(index*dim + a_ind)
                        mass_cols.append(jndex*dim + b_ind)
                        mass_data.append(acc)
        

        mass = csr_matrix((mass_data, (mass_rows, mass_cols)), shape=(dim*N, dim*N))      
        self._vector_field_mass_matrix = .5*(mass + mass.T)

#______________________________________________________
#______________________________________________________
#______________________________________________________
# HOF matrix constructions level methods

    def _compute_function_stiffness_matrix(self, data, func):

        # resolve dependencies 
        if self._local_mesh is None:
            self._compute_local_mesh(data)

        if self._coefs is None:
            self._compute_mls_coefs(data)

        u_1, u_2 = self._get_triangle_grid()    # constants
        N = data.shape[0]

        stiffness_rows, stiffness_cols, stiffness_data = [], [], []     # store data for csr_matrix

        # construct rows of stiffness matrix 
        for index in range(N):

            polynomial = self._coefs[index]
            neighborhood = self._coordinates[index]
            [_, first_ring, vertices] = self._local_mesh[index]     # row-wise constants

            # compute off diagonals first (incase diagonalization)
            for vertex in vertices:
                jndex = self._knn_indices[index, vertex]                
                triangle_pair = first_ring[(first_ring == vertex).any(axis = 1),:]
                
                acc = 0
                for triangle in triangle_pair:
                    triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 0 if x == vertex else 1)
                    points = neighborhood[triangle, :]          # [base_point, vertex, other]

                    acc += func(polynomial, points, u_1, u_2)     # compute integral 

            
                stiffness_rows.append(index)               
                stiffness_cols.append(jndex)
                stiffness_data.append(acc)          # store the integral we just computed
               

        # if a strategy isn't set, compute diagonals analytically 
        if self._stiffness_diagonalization is None:
            for index in range(N):
                neighborhood = self._coordinates[index]
                [_, first_ring, vertices] = self._local_mesh[index]
                polynomial = self._coefs[index]

                acc = 0
                for triangle in first_ring:
                    triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 1)
                    points = neighborhood[triangle]

                    g_inv = self._riemannian_metric_inverse(polynomial, points, u_1, u_2)
                    integrand = g_inv.sum(axis=2).sum(axis=1)
                    acc += integrand.mean() 
                
                stiffness_rows.append(index)          
                stiffness_cols.append(index)
                stiffness_data.append(acc)
        

        A = csr_matrix((stiffness_data, (stiffness_rows, stiffness_cols)), shape=(N, N)) 

        # if we didn't compute diagonals above, do it now using the strategy
        if self._stiffness_diagonalization is not None:
            return self._diagonalize_stiffness(A)     
        else:
            return .5*(A + A.T)
        
    def _compute_vector_field_stiffness_matrix(self, data, func):

        # resolve dependencies 
        if self._local_mesh is None:
            self._compute_local_mesh(data)

        if self._coefs is None:
            self._compute_mls_coefs(data)

        dim = self.dim
        N = data.shape[0]
        u_1, u_2 = self._get_triangle_grid()        # constants


        stiffness_rows, stiffness_cols, stiffness_data = [],[],[]

        # construct each row 
        for index in range(N):
            Basis_i, _ = self._compute_nodal_basis(index, index)
            neighborhood, polynomial = self._coordinates[index], self._coefs[index]
            [_, first_ring, vertices] = self._local_mesh[index]          # row-wise constants

            for a_ind in range(dim):
                for b_ind in range(dim):

                    acc = 0
                    for triangle in first_ring:
                        triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 1)
                        points = neighborhood[triangle, :]
                        
                        acc += func(polynomial, points, 0, Basis_i[a_ind], Basis_i[b_ind], u_1, u_2).mean()
                    
                    stiffness_rows.append(index*dim + a_ind)
                    stiffness_cols.append(index*dim + b_ind)
                    stiffness_data.append(acc)
                

            # now, loop through off diagonal triangle pairs
            for vertex in vertices:
                jndex = self._knn_indices[index, vertex]
                Basis_i, Basis_j = self._compute_nodal_basis(index, jndex)

                triangle_pair = first_ring[(first_ring == vertex).any(axis = 1),:]  # identify pair of triangles with x_i x_j as a side 
                for a_ind in range(dim):
                    for b_ind in range(dim):
        
                        acc = 0
                        for triangle in triangle_pair:
                            triangle = sorted(triangle, key = lambda x: -1 if x == 0 else 0 if x == vertex else 1)      # sort so partial u_1 points from p_0 to vertex
                            points = neighborhood[triangle, :]
                            
                            acc += func(polynomial, points, vertex, Basis_i[a_ind], Basis_j[b_ind], u_1, u_2).mean()
                        
                        stiffness_rows.append(index*dim + a_ind)
                        stiffness_cols.append(jndex*dim + b_ind)
                        stiffness_data.append(acc)
                
                        
        stiffness = csr_matrix((stiffness_data, (stiffness_rows, stiffness_cols)), shape=(dim*N, dim*N))      
        return .5*(stiffness + stiffness.T)

#______________________________________________________
#______________________________________________________
#______________________________________________________

# mls and mesh level methods                  
#  
    def _compute_local_mesh(self, data):

        # resolve dependency
        if self._coordinates is None:
            self._construct_local_coodinates(data)
        
        N = data.shape[0]

        local_mesh = {}     # initialize for constructing 
        for i in range(N):
            neighborhood = self._coordinates[i]

            tri = Delaunay(neighborhood).simplices
            first_ring = tri[(tri == 0).any(axis = 1),:]
            vertices = unique(first_ring)
            vertices = vertices[vertices != 0]      # basepoint will not be in vertices 

            local_mesh[i] = [tri, first_ring, vertices]
        
        self._local_mesh = local_mesh

    def _compute_mls_coefs(self, data):
        if self._coordinates is None:
            self._construct_local_coodinates(data)
        
        N, dim, n_neighbors = data.shape[0], self.dim, self.n_neighbors     # constants

        # 1, x, y, x^2, xy, y^2
        poly = PolynomialFeatures(dim)      # dummy fit to get number of output features (L)
        poly.fit_transform([[0 for _ in range(dim)]])
        L = poly.n_output_features_

        self._coefs = zeros((N, L))         # for storing polynomial coefficients 

        for i in range(N):
            X, y = self._coordinates[i,:,:], self._mls_target[i,:]
            X = poly.fit_transform(X)

            W = diag(1/n_neighbors*ones((n_neighbors,)))
            W[0,0] = 1

            A = pinv(X.T @ W @ X) @ X.T @ W

            self._coefs[i,:] = (A @ y).reshape(-1)

    def _construct_local_coodinates(self, data):
        # resolve dependency
        if self._knn_indices is None:
            self._compute_knn(data)

        N, n, n_neighbors, dim = data.shape[0], data.shape[1], self.n_neighbors, self.dim   

        local_data, coordinates = data[self._knn_indices], zeros((N,n_neighbors,n))     # for storing data 
        Proj = self._construct_tangential_projection(local_data)              

        for i in range(N):
            self._tangent_basis[i] = [Proj[i][:,j] for j in range(dim)]
            neighborhood = (local_data[i] - local_data[i][0,:])
            coordinates[i] = neighborhood @ Proj[i]
        
        self._coordinates = coordinates[:,:,0:dim]
        self._mls_target = coordinates[:,:,dim:]

    def _compute_knn(self, data):
        N, n, n_neighbors = data.shape[0], data.shape[1], self.n_neighbors
        tree = KDTree(data)
        _, indices = tree.query(x = data, k = n_neighbors)
        self._knn_indices = indices

    def _construct_tangential_projection(self, local_data):
        N, n_neighbors, n = local_data.shape    # constants

        Proj = zeros((N,n,n))  
        for i in range(local_data.shape[0]):
            neighborhood = local_data[i]
            barycenter = (1/n_neighbors)*neighborhood.sum(axis = 0)
            neighborhood = neighborhood - barycenter    # center local neighborhood

            P = zeros((n,n))
            for j in range(n_neighbors):
                p = neighborhood[j,:].reshape((n,1))
                P = P + p @ p.T             # computing local covariance matrix
            

            eigenValues, eigenVectors = eigh(P)     # use eigenvectors to project 
            idx = eigenValues.argsort()[::-1]   
            eigenVectors = eigenVectors[:,idx]

            Proj[i] = eigenVectors[:,0:n]

        
        return Proj

#______________________________________________________
#______________________________________________________
#______________________________________________________
# triangle level geometry formulas
    
    # integrands 
    def _vector_field_mass_integrand(self, polynomial, # tuple of coefficients for polynomial 
                                            points,       # points in T_PM that makeup the triangle 
                                            vertex,       # detect if diagonal block (vertex == 0) for off diagonal
                                            e_a, e_b,     # e_a is first or second eigenvector of P_i, e_b is of P_j - n x 1 vectors 
                                        u_1, u_2):
        
        # regress e_a and e_b onto barycentric coordiantes
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0
        A = zeros((2,2))
        A[:,0] = p_1 
        A[:,1] = p_2

        g = self._riemannian_metric(polynomial, points, u_1, u_2)       # columns of A are partial_1, partial_2

        a_1, a_2 = lstsq(A, e_a, rcond = None)[0]
        b_1, b_2 = lstsq(A, e_b, rcond = None)[0]
        

        # triangle specific quantities 
        det_g = g[:,0,0]*g[:,1,1] - g[:,0,1]*g[:,1,0]
        hat = 1 - u_1 - u_2


        # case i == j
        if vertex == 0:
            integrand = (a_1*b_1*g[:,0,0] + a_1*b_2*g[:,0,1] + a_2*b_1*g[:,1,0] + a_2*b_2*g[:,1,1]) * hat * hat * sqrt(det_g)
        # case i != j
        else:
            integrand = (a_1*b_1*g[:,0,0] + a_1*b_2*g[:,0,1] + a_2*b_1*g[:,1,0] + a_2*b_2*g[:,1,1]) * hat * u_1 * sqrt(det_g)

        return integrand
    
    def _bochner_stiffness_integrand(self, polynomial, 
                                           points, 
                                           vertex, 
                                           e_a, e_b, 
                                           u_1, u_2):
        N = u_1.shape[0]
        # regress e_a and e_b onto the barycentric coordinates
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0
        A = zeros((2,2))
        A[:,0] = p_1 
        A[:,1] = p_2

        g = self._riemannian_metric(polynomial, points, u_1, u_2)       # columns of A are partial_1, partial_2

        a_1, a_2 = lstsq(A, e_a, rcond = None)[0]
        b_1, b_2 = lstsq(A, e_b, rcond = None)[0]


        # compute mls quantities  
        hessian = self._polynomial_embedding_hessian(polynomial)
        gradient = self._polynomial_embedding_gradient(polynomial, points, u_1, u_2)

        # compute triangle specific quantities 
        det_g = g[:,0,0]*g[:,1,1] - g[:,0,1]*g[:,1,0]
        g_inv = self._riemannian_metric_inverse(polynomial, points, u_1, u_2)
        Gamma_1 = self._christoffel_matrix(points, hessian, gradient, g_inv, 0)
        Gamma_2 = self._christoffel_matrix(points, hessian, gradient, g_inv, 1)

        hat = 1 - u_1 - u_2                     # hat function and its tensorized version (for scaling tensors)
        vHat = hat.repeat(4).reshape((N, 2, 2))

        a = (vHat*Gamma_1 + array([[-1, -1], [0, 0]])) @ g_inv       # locally, grad_g ( hat partial_1 ) <-- a (2,0) tensor field
        b = (vHat*Gamma_2 + array([[0, 0], [-1, -1]])) @ g_inv       # locally, grad_g ( hat partial_2 ) <-- a (2,0) tensor field


        # case i == j <-- compute integrand 
        if vertex == 0:     

            c_11 = trace(a @ g @ transpose(a, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_12 = trace(a @ g @ transpose(b, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_21 = trace(b @ g @ transpose(a, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_22 = trace(b @ g @ transpose(b, axes = (0,2,1)) @ g, axis1=1, axis2=2)

            integrand = (a_1*b_1*c_11 + a_1*b_2*c_12 + a_2*b_1*c_21 + a_2*b_2*c_22)*sqrt(det_g)

        # case i != j <-- compute integrand 
        else:              
            vU_1 = u_1.repeat(4).reshape((N, 2, 2))

            c = (vU_1*Gamma_1 +  array([[1, 0], [0, 0]])) @ g_inv
            d = (vU_1*Gamma_2 +  array([[0, 0], [1, 0]])) @ g_inv

            c_11 = trace(a @ g @ transpose(c, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_12 = trace(a @ g @ transpose(d, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_21 = trace(b @ g @ transpose(c, axes = (0,2,1)) @ g, axis1=1, axis2=2)
            c_22 = trace(b @ g @ transpose(d, axes = (0,2,1)) @ g, axis1=1, axis2=2)

            integrand = (a_1*b_1*c_11 + a_1*b_2*c_12 + a_2*b_1*c_21 + a_2*b_2*c_22)*sqrt(det_g)
        
        return integrand

    def _hodge_stiffness_integrand(self, polynomial, points, vertex, e_a, e_b, u_1, u_2):
        N = u_1.shape[0]
        # regress e_a and e_b onto the barycentric coordinates
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0
        A = zeros((2,2))
        A[:,0] = p_1 
        A[:,1] = p_2

        c_1, c_2 = lstsq(A, e_a, rcond = None)[0]
        d_1, d_2 = lstsq(A, e_b, rcond = None)[0]


        # compute mls quantities  
        hat = (1-u_1-u_2)
        g = self._riemannian_metric(polynomial, points, u_1, u_2) 

        hessian = self._polynomial_embedding_hessian(polynomial)
        gradient = self._polynomial_embedding_gradient(polynomial, points, u_1, u_2)
        partial_g = {
            (0,0,0): self._riemannian_metric_derivative(points, hessian, gradient, 0,0,0),
            (0,0,1): self._riemannian_metric_derivative(points, hessian, gradient, 0,0,1),

            (0,1,0): self._riemannian_metric_derivative(points, hessian, gradient, 0,1,0),
            (0,1,1): self._riemannian_metric_derivative(points, hessian, gradient, 0,1,1),

            (1,1,0): self._riemannian_metric_derivative(points, hessian, gradient, 1,1,0),
            (1,1,1): self._riemannian_metric_derivative(points, hessian, gradient, 1,1,1)    
        }

        sqrt_det_g = sqrt(det(g))

        # \partial (sqrt(det(g)))
        partial_sqrt_det_g = {
            0: ( 1 / (2*sqrt_det_g) ) * ( g[:,1,1]*partial_g[0,0,0] + g[:,0,0]*partial_g[1,1,0] - 2*g[:,0,1]*partial_g[0,1,0]), 
            1: ( 1 / (2*sqrt_det_g) ) * ( g[:,1,1]*partial_g[0,0,1] + g[:,0,0]*partial_g[1,1,1] - 2*g[:,0,1]*partial_g[0,1,1])
            }

        A_1 = g[:,0,0] - g[:,0,1] + hat*(partial_g[0,1,0] - partial_g[0,0,1])
        A_2 = g[:,0,1] - g[:,1,1] + hat*(partial_g[1,1,0] - partial_g[0,1,1])
        A_3 = 1 - (hat/sqrt_det_g)*partial_sqrt_det_g[0]
        A_4 = 1 - (hat/sqrt_det_g)*partial_sqrt_det_g[1]

        if vertex == 0:
            part_1 = (1/sqrt_det_g) * (c_1*d_1*A_1*A_1 + c_1*d_2*A_1*A_2 + c_2*d_1*A_1*A_2 + c_2*d_2*A_2*A_2) 
            part_2 = sqrt_det_g * (c_1*d_1*A_3*A_3 + c_1*d_2*A_3*A_4 + c_2*d_1*A_3*A_4 + c_2*d_2*A_4*A_4)
            integrand = part_1 + part_2


        else:
            B_1 = g[:,1,0] + u_1*(partial_g[0,1,0] - partial_g[0,0,1])
            B_2 = g[:,1,1] + u_1*(partial_g[1,1,0] - partial_g[0,1,1])
            B_3 = -1 - (u_1/sqrt_det_g)*partial_sqrt_det_g[0]
            B_4 = -(u_1/sqrt_det_g)*partial_sqrt_det_g[1]

            part_1 = (1/sqrt_det_g) * (c_1*d_1*A_1*B_1 + c_1*d_2*A_1*B_2 + c_2*d_1*B_1*A_2 + c_2*d_2*A_2*B_2) 
            part_2 = sqrt_det_g * (c_1*d_1*A_3*B_3 + c_1*d_2*A_3*B_4 + c_2*d_1*B_3*A_4 + c_2*d_2*A_4*B_4)
            integrand = part_1 + part_2


        return integrand



    # metric formulas 
    def _riemannian_metric(self, polynomial, points, u_1, u_2):          # barycentric coordinates
        N = u_1.shape[0]

        _, d, e, a, c, b = polynomial     # ax^2 + by^2 + cxy + dx + ey + bias
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0

        xx = outer(u_1, p_1) + outer(u_2, p_2)

        x,y = xx[:,0], xx[:,1]

        nabla = stack((2*a*x + c*y + d, 2*b*y + c*x + e), axis = 1)


        q = dot(p_1, p_1) + (nabla @ p_1)**2
        r = dot(p_1, p_2) + (nabla @ p_1)*(nabla @ p_2)
        s = dot(p_2, p_2) + (nabla @ p_2)**2

        g = zeros((N, 2, 2))

        g[:,0,0] = q
        g[:,0,1] = r 
        g[:,1,0] = r
        g[:,1,1] = s


        return g 

    def _riemannian_metric_inverse(self, polynomial, points, u_1, u_2):
        g = self._riemannian_metric(polynomial,points, u_1, u_2)
        det_g = det(g)
        N = g.shape[0]

        metric_inv = zeros((N,2,2))

        metric_inv[:,0,0] =  g[:,1,1]/det_g
        metric_inv[:,0,1] = -g[:,1,0]/det_g
        metric_inv[:,1,0] = -g[:,0,1]/det_g
        metric_inv[:,1,1] = g[:,0,0]/det_g

        return metric_inv

    def _riemannian_metric_derivative(self, points, hessian, gradient, i,j,k): # partial_k (g_{ij}) as a function of (u_1, u_2)
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0

        partial_dict = {0: p_1, 1: p_2}

        p_i = partial_dict[i]
        p_j = partial_dict[j]
        p_k = partial_dict[k]               # gradient is N X 2

        return dot(hessian @ p_k, p_j) * (gradient @ p_i) + dot(hessian @ p_k, p_i) * (gradient @ p_j)

    def _christoffel_symbols(self, points, hessian, gradient, metric_inv, i,k,l):

        g_inv_i1 = metric_inv[:,i,0]
        g_inv_i2 = metric_inv[:,i,1]

        a_1 = self._riemannian_metric_derivative(points, hessian, gradient, 0,k,l)
        a_2 = self._riemannian_metric_derivative(points, hessian, gradient, 1,k,l)

        b_1 = self._riemannian_metric_derivative(points, hessian, gradient, 0,l,k)
        b_2 = self._riemannian_metric_derivative(points, hessian, gradient, 1,l,k)

        c_1 = self._riemannian_metric_derivative(points, hessian, gradient, k,l,0)
        c_2 = self._riemannian_metric_derivative(points, hessian, gradient, k,l,1)

        return (1/2)*g_inv_i1*(a_1 + b_1 - c_1) + (1/2)*g_inv_i2*(a_2 + b_2 - c_2)

    def _christoffel_matrix(self, points, hessian, gradient, metric_inv, j):
        N = gradient.shape[0]

        Gamma = zeros((N,2,2))

        Gamma[:,0,0] = self._christoffel_symbols(points, hessian, gradient, metric_inv, 0,j,0)
        Gamma[:,0,1] = self._christoffel_symbols(points, hessian, gradient, metric_inv, 0,j,1)
        Gamma[:,1,0] = self._christoffel_symbols(points, hessian, gradient, metric_inv, 1,j,0)
        Gamma[:,1,1] = self._christoffel_symbols(points, hessian, gradient, metric_inv, 1,j,1)

        return Gamma

#______________________________________________________
#______________________________________________________
#______________________________________________________
# local mesh level formulas 
    
    def _polynomial_embedding_gradient(self, polynomial, points, u_1, u_2):
        _, d, e, a, c, b = polynomial     # ax^2 + by^2 + cxy + dx + ey + bias
        p_0, p_1, p_2 = points[0,:], points[1,:], points[2,:]
        p_1, p_2 = p_1 - p_0, p_2 - p_0

        xx = outer(u_1, p_1) + outer(u_2, p_2)

        x,y = xx[:,0], xx[:,1]

        nabla = stack((2*a*x + c*y + d, 2*b*y + c*x + e), axis = 1)

        return nabla

    def _polynomial_embedding_hessian(self, polynomial):
        _, _, _, a, c, b = polynomial     # ax^2 + by^2 + cxy + dx + ey + bias


        hessian = array([[2*a, c], [c, 2*b]])

        return hessian
  
    def _compute_nodal_basis(self, index, jndex):
        if index == jndex:
            return (array([1,0]), array([0,1])), (array([1,0]), array([0,1]))

        e_1 = array([1,0])
        e_2 = array([0,1])

        B = zeros((2, 2)) # (n, d)
        B[:,0], B[:,1] = e_1, e_2


        A = stack(self._tangent_basis[index], axis = 1)
        b = stack(self._tangent_basis[jndex], axis = 1)

        coefs, residuals, _, _ = lstsq(A, b, rcond=None)

        V = B @ coefs 
        e_3, e_4 = V[:,0], V[:,1]
        
        return (e_1, e_2), (e_3, e_4)

#______________________________________________________
#______________________________________________________
#______________________________________________________
# utility 
   
    def _get_triangle_grid(self,):
        N = self._grid_size 
        if N == 3:
            data = array([[0,0], [1,0], [0,1]])

            return data[:,0], data[:,1]
        
        else:
            h_x = 1/(2*N)
            h_y = 1/(2*N)

            data = zeros((2*N,2*N,2))

            for i in range(2*N):
                for j in range(2*N):
                    data[i,j,0] = i*h_x
                    data[i,j,1] = j*h_y

            data = data.reshape(-1,2)
            mask = (data[:,0] + data[:,1] <= 1)
            data = data[mask]

            return data[:,0], data[:,1] 

    def _diagonalize_stiffness(self, A):
        N = A.shape[0]

        if self._stiffness_diagonalization == "uniform-data":
            S = .5*(A + A.T)
            for i in range(N):
                S[i,i] = 0
                row_sum = S[i,:].sum()
                S[i,i] = -row_sum

        elif self._stiffness_diagonalization == "random-data":
            S = A.copy()

            for i in range(N):
                # set off diagonals
                for j in range(i+1,N):      # only these needed since it is symmetric
                    case_1 = (A[i,j] <= 0) and (A[j,i] <= 0)

                    if case_1:
                        S[i,j] = max(A[i,j], A[j,i])
                    else:
                        S[i,j] = min(A[i,j], A[j,i])
                    
                    S[j,i] = S[i,j] # ensure it's symmetric 
            
                row_sum = S[i,:]

                # set diagonals 
                S[i,i] = -row_sum.sum()
        
        else:
            error_message = f"Invalid _stiffness_diagonalization: {self._stiffness_diagonalization} is not in {self.__diagonalization_strategies}"
            raise KeyError(error_message)
        

        return S