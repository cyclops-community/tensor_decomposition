import numpy as np
import queue
from .common_kernels import solve_sys, compute_lin_sysN

class DTALS_Optimizer(object):

    def __init__(self,tenpy,T,A):
        self.tenpy = tenpy
        self.T = T
        self.A = A

    def einstr_builder(self,M,s,ii):
        ci = ""
        nd = M.ndim
        if len(s) != 1:
            ci ="R"
            nd = M.ndim-1

        str1 = "".join([chr(ord('a')+j) for j in range(nd)])+ci
        str2 = (chr(ord('a')+ii))+"R"
        str3 = "".join([chr(ord('a')+j) for j in range(nd) if j != ii])+"R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def step(self,Regu):
        q = queue.Queue()
        for i in range(len(self.A)):
            q.put(i)
        s = [(list(range(len(self.A))),self.T)]
        while not q.empty():
            i = q.get()
            while i not in s[-1][0]:
                s.pop()
                assert(len(s) >= 1)
            while len(s[-1][0]) != 1:
                M = s[-1][1]
                idx = s[-1][0].index(i)
                ii = len(s[-1][0])-1
                if idx == len(s[-1][0])-1:
                    ii = len(s[-1][0])-2
                
                einstr = self.einstr_builder(M,s,ii)

                N = self.tenpy.einsum(einstr,M,self.A[ii])
                ss = s[-1][0].copy()
                ss.remove(ii)
                s.append((ss,N))
            self.A[i] = solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), s[-1][1])
        return self.A



class PPALS_Optimizer(DTALS_Optimizer):
    """Pairwise perturbation CP decomposition optimizer

    Attributes:
        pp (bool): using pairwise perturbation or dimension tree to update.
        reinitialize_tree (bool): reinitialize the dimension tree or not.
        tol_restart_dt (float): tolerance for restarting dimention tree.
        tree (dictionary): store the PP dimention tree.
        order (int): order of the input tensor.
        dA (list): list of perturbation terms.

    References:
        Linjian Ma and Edgar Solomonik; Accelerating Alternating Least Squares for Tensor Decomposition 
        by Pairwise Perturbation; arXiv:1811.10573 [math.NA], November 2018.
    """

    def __init__(self,tenpy,T,A,tol_restart_dt):
        super(PPALS_Optimizer, self).__init__(tenpy,T,A)
        self.pp = False
        self.reinitialize_tree = False
        self.tol_restart_dt = tol_restart_dt
        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.order = len(A)
        self.dA = []  
        for i in range(order):
            self.dA.append(tenpy.zeros((self.T.shape[i],R)))

    def _get_nodename(self, nodeindex):
        """Based on the index, output the node name used for the key of self.tree.

        Args:
            nodeindex (numpy array): A numpy array containing the indexes that
                the input tensor is not contracted with.

        Returns:
            (string) A string correspoding to the input array. 
        
        Example:
            When the input tensor has 4 dimensions:
            _get_nodename(np.array([1,2])) == 'bc'

        """
        if len(nodeindex) == order:
            return '0'
        return "".join([chr(ord('a')+j) for j in range(nodeindex)])

    def _get_einstr(self, nodeindex, parent_nodeindex, contract_index):
        """Build the Einstein string for the contraction. 

        This function contract the tensor represented by the parent_nodeindex and 
        the matrix represented by the contract_index and output the string.

        Args:
            nodeindex (numpy array): represents the contracted tensor.
            parent_nodeindex (numpy array): represents the contracting tensor.
            contract_index (int): index in self.A

        Returns:
            (string) A string used in self.tenpy.einsum
        
        Example:
            When the input tensor has 4 dimensions:
            _get_einstr(np.array([1,2]), np.array([1,2,3]), 3) == "abcR,cR->abR"

        """
        ci = ""
        if len(parent_nodeindex) != order:
            ci = "R"

        str1 = "".join([chr(ord('a')+j) for j in parent_nodeindex]) + ci
        str2 = (chr(ord('a')+contract_index)) + "R"
        str3 = "".join([chr(ord('a')+j) for j in nodeindex]) + "R"
        einstr = str1 + "," + str2 + "->" + str3
        return einstr

    def _get_parentnode(self, nodeindex):
        """Get the parent node based on current node index 

        Args:
            nodeindex (numpy array): represents the current tensor.

        Returns:
            parent_nodename (string): representing the key of parent node in self.tree 
            parent_index (numpy array): the index of parent node
            contract_index (int): the index difference between the current and parent node
        
        """
        fulllist = np.array(range(order))

        comp_index = np.setdiff1d(fulllist, nodeindex)
        comp_parent_index = comp_index[:-1]

        contract_index = comp_index[-1]
        parent_index = np.setdiff1d(fulllist, comp_parent_index)
        parent_nodename = _get_nodename(parent_index)

        return parent_nodename, parent_index, contract_index

    def _initialize_treenode(self, nodeindex):
        """Initialize one node in self.tree

        Args:
            nodeindex (numpy array): The target node index
        
        """
        nodename = self._get_nodename(nodeindex)
        parent_nodename, parent_nodeindex, contract_index = self._get_parentnode(nodeindex)
        einstr = self._get_einstr(nodeindex,parent_nodeindex,contract_index)

        if not self.tree.find(parent_nodename):
            self._initialize_treenode(parent_nodeindex)

        N = self.tenpy.einsum(einstr,self.tree[parent_nodename][1],self.A[contract_index])

        self.tree[nodename] = (nodeindex,N)

    def _initialize_tree(self):
        """Initialize self.tree       
        """

        self.tree = { '0':(list(range(len(self.A))),self.T) }
        self.dA = []  
        for i in range(order):
            self.dA.append(tenpy.zeros((self.T.shape[i],R)))

        for ii in range(0, order):
            for jj in range(ii+1, order):
                self._initialize_treenode(np.array([ii,jj]))

        for ii in range(0, order):
            self._initialize_treenode(np.array([ii]))

    def _step_pp_subroutine(self,Regu):
        """Doing one step update based on pairwise perturbation

        Args:
            Regu (float): Regularization term

        Returns:
            A (list): list of decomposed matrices
        
        """
        for i in range(order):
            nodename = self._get_nodename(np.array([i]))
            N = self.tree[nodename]

            for j in range(i):
                parentname = self._get_nodename(np.array([j,i]))
                einstr = self._get_einstr(np.array([i]), np.array([j,i]), np.array([j]))
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])
            for j in range(i+1, order):
                parentname = self._get_nodename(np.array([i,j]))
                einstr = self._get_einstr(np.array([i]), np.array([i,j]), np.array([j]))
                N += self.tenpy.einsum(einstr,self.tree[parentname][1],self.dA[j])

            output = solve_sys(self.tenpy,compute_lin_sysN(self.tenpy,self.A,i,Regu), N)
            self.dA[i] = output - self.A[i]
            self.A = output

        num_smallupdate = 0
        for i in range(order):
            if tenpy.sum(self.dA[i]**2,axis=0)**.5 > self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate > 0:
            self.pp = False
            self.reinitialize_tree = False

        return self.A


    def _step_dt_subroutine(self,Regu):
        """Doing one step update based on dimension tree

        Args:
            Regu (float): Regularization term

        Returns:
            A (list): list of decomposed matrices
        
        """
        A_update = super(PPALS_Optimizer, self).step(Regu)
        num_smallupdate = 0
        for i in range(order):
            self.dA[i] = A_update[i] - self.A[i]
            if tenpy.sum(self.dA[i]**2,axis=0)**.5 < self.tol_restart_dt:
                num_smallupdate += 1

        if num_smallupdate == order:
            self.pp = True
            self.reinitialize_tree = True
        return A

    def step(self,Regu):
        """Doing one step update in the optimizer

        Args:
            Regu (float): Regularization term

        Returns:
            A (list): list of decomposed matrices
        
        """
        if self.pp:
            if self.reinitialize_tree:
                self._initialize_tree()
                self.reinitialize_tree = False
            A = self._step_pp_subroutine(Regu)
        else:
            A = self._step_dt_subroutine(Regu)
        return A
